import numpy as np
from skimage.color import rgb2gray

from tqdm import tqdm
from tadataka.coordinates import image_coordinates
from tadataka.dataset import TumRgbdDataset
from tadataka.vo.semi_dense import SemiDenseVO
from tadataka.vo.semi_dense.frame import Frame
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.projection import pi, Warp
from tadataka.matrix import to_homogeneous
from tadataka.rigid_transform import transform
from tests.dataset.path import new_tsukuba

from matplotlib import pyplot as plt


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


dataset = TumRgbdDataset(
    "datasets/rgbd_dataset_freiburg1_desk2",
    which_freiburg=1
)

from tadataka.vo.semi_dense.semi_dense import InverseDepthEstimator

kf = dataset[0]
rf = dataset[5]

# x, y = 0, 100
# u_key = np.array([x, y])
# plot = PlotWarp(kf.image, rf.image,
#                 kf.camera_model, rf.camera_model,
#                 kf.pose, rf.pose)
# plot(u_key, kf.depth_map[y, x])

keyframe = Frame(kf.camera_model, rgb2gray(kf.image), kf.pose)
refframe = Frame(rf.camera_model, rgb2gray(rf.image), rf.pose)

from tadataka.rigid_transform import Warp3D
estimator = InverseDepthEstimator(
    keyframe,
    sigma_i=0.01, sigma_l=0.02,
    step_size_ref=0.01, min_gradient=1.0
)
image_shape = keyframe.image.shape
prior = invert_depth(kf.depth_map)

inv_depth_map = np.empty(image_shape)
for u_key in tqdm(image_coordinates(image_shape)):
    x, y = u_key

    inv_depth_map[y, x] = estimator(refframe, u_key, prior[y, x], prior[y, x])
    if False:
        warp = Warp3D(keyframe.pose, refframe.pose)
        x_key = keyframe.camera_model.normalize(u_key)
        x_ref = pi(warp(invert_depth(prior[y, x]) * to_homogeneous(x_key)))
        u_ref = refframe.camera_model.unnormalize(x_ref)

        fig = plt.figure()

        ax = fig.add_subplot(121)
        ax.set_title("keyframe")
        ax.imshow(keyframe.image, cmap="gray")
        ax.scatter(u_key[0], u_key[1], c="red")

        ax = fig.add_subplot(122)
        ax.set_title("reference frame")
        ax.imshow(refframe.image, cmap="gray")
        ax.scatter(u_ref[0], u_ref[1], c="red")
        plt.show()

fig = plt.figure()

ax = fig.add_subplot(131)
ax.set_title("keyframe")
ax.imshow(keyframe.image, cmap="gray")

ax = fig.add_subplot(132)
ax.set_title("inverse depth map")
ax.imshow(inv_depth_map, cmap="gray")

ax = fig.add_subplot(133)
ax.set_title("depth map")
ax.imshow(invert_depth(inv_depth_map), cmap="gray")

plt.show()

# estimator = InverseDepthMapEstimator(kf, [rf], pixel_age)
# inv_depth_map, variance_map = estimator(prior_inv_depth_map, prior_variance_map)
# vo = SemiDenseVO()
# vo.estimate(refframe)
# vo.estimate(keyframe)
#
# for frame, _ in dataset:
#     vo.estimate(frame)
