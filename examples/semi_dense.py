import numpy as np
from skimage.color import rgb2gray, gray2rgb

from tqdm import tqdm
from tadataka.coordinates import image_coordinates
from tadataka.dataset import TumRgbdDataset
from tadataka.vo.semi_dense import SemiDenseVO
from tadataka.vo.semi_dense.frame import Frame
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.projection import pi, Warp
from tadataka.matrix import to_homogeneous
from tadataka.rigid_transform import transform
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tests.dataset.path import new_tsukuba

import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def replace_nan(array, value):
    array[np.isnan(array)] = value
    return array


def as_uint8(array):
    return np.round(array, 0).astype(np.uint8)


def scale(image, srange, trange):
    smin, smax = srange
    tmin, tmax = trange
    ratio = (tmax - tmin) / (smax - smin)
    return ratio * (image - smin) + tmin


def flag_to_color(flag):
    if flag == FLAG.SUCCESS:
        return np.array([0, 1, 0])  # green
    if flag == FLAG.KEY_OUT_OF_RANGE:
        return np.array([0, 0, 0])  # black
    if flag == FLAG.EPIPOLAR_TOO_SHORT:
        return np.array([1, 0, 0])  # red
    if flag == FLAG.INSUFFICIENT_GRADIENT:
        return np.array([0, 0, 1])  # blue
    raise ValueError


class PlotWarp(object):
    def __init__(self, image_key, image_ref,
                 camera_model_key, camera_model_ref,
                 pose_key, pose_ref):
        self.image_key = image_key
        self.image_ref = image_ref
        self.warp = Warp(camera_model_key, camera_model_ref,
                         pose_key, pose_ref)

    def __call__(self, u_key, depth_key):
        u_ref = self.warp(u_key, depth_key)

        fig = plt.figure()

        ax = fig.add_subplot(121)
        ax.set_title("keyframe")
        ax.imshow(self.image_key)
        ax.scatter(u_key[0], u_key[1], c="red")

        ax = fig.add_subplot(122)
        ax.set_title("reference frame")
        ax.imshow(self.image_ref)
        ax.scatter(u_ref[0], u_ref[1], c="red")

        plt.show()


def plot_with_bar(ax, image, vrange):
    vmin, vmax = vrange
    im = ax.imshow(image, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


dataset = TumRgbdDataset(
    "datasets/rgbd_dataset_freiburg1_desk",
    which_freiburg=1
)

from tadataka.vo.semi_dense.semi_dense import InverseDepthEstimator

kf = dataset[0]
rf = dataset[5]

keyframe = Frame(kf.camera_model, rgb2gray(kf.image), kf.pose)
refframe = Frame(rf.camera_model, rgb2gray(rf.image), rf.pose)

from tadataka.rigid_transform import Warp3D
estimator = InverseDepthEstimator(
    keyframe,
    sigma_i=0.01, sigma_l=0.02,
    step_size_ref=0.005, min_gradient=10.0
)
image_shape = keyframe.image.shape

prior_depth = np.random.uniform(0.8, 1.2, size=kf.depth_map.shape)
prior_variance = 1.0 * np.ones(image_shape)

depths_pred = []
flag_map = np.zeros((*image_shape, 3))
for u_key in tqdm(image_coordinates(image_shape)):
    x, y = u_key
    inv_depth, flag = estimator(refframe, u_key,
                                prior_depth[y, x], prior_variance[y, x])
    if flag == FLAG.SUCCESS:
        depths_pred.append([x, y, invert_depth(inv_depth)])
    flag_map[y, x] = flag_to_color(flag)

depths_pred = np.array(depths_pred)

fig = plt.figure()

ax = fig.add_subplot(231)
ax.set_title("keyframe")
ax.imshow(kf.image)

ax = fig.add_subplot(232)
ax.set_title("reference frame")
ax.imshow(rf.image)

vmin = min(np.nanmin(kf.depth_map), np.nanmin(depths_pred[:, 2]))
vmax = max(np.nanmax(kf.depth_map), np.nanmax(depths_pred[:, 2]))

trange = [0.0, 1.0]

ax = fig.add_subplot(234)
ax.set_title("flag map")
ax.imshow(flag_map)
patches = [Patch(facecolor=flag_to_color(f), label=f.name) for f in FLAG]
ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05))

cmap = 'RdBu'

ax = fig.add_subplot(235)
ax.set_title("predicted depth map")
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
mapper = ScalarMappable(norm=norm, cmap=cmap)
ax.imshow(kf.image)
ax.scatter(depths_pred[:, 0], depths_pred[:, 1],
           s=1.0, c=mapper.to_rgba(depths_pred[:, 2]))

ax = fig.add_subplot(236)
ax.set_title("ground truth")
im = ax.imshow(kf.depth_map, norm=norm, cmap=cmap)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

plt.show()


# x, y = 250, 250
# u_key = np.array([x, y])
# plot = PlotWarp(kf.image, rf.image,
#                 kf.camera_model, rf.camera_model,
#                 kf.pose, rf.pose)
# plot(u_key, depth_map_pred[y, x])

# estimator = InverseDepthMapEstimator(kf, [rf], pixel_age)
# inv_depth_map, variance_map = estimator(prior_inv_depth_map, prior_variance_map)
# vo = SemiDenseVO()
# vo.estimate(refframe)
# vo.estimate(keyframe)
#
# for frame, _ in dataset:
#     vo.estimate(frame)
