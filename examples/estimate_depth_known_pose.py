import numpy as np
from skimage.color import rgb2gray
from tadataka.pose import WorldPose
from tadataka.warp import warp2d, Warp2D
from tadataka.dataset import NewTsukubaDataset, TumRgbdDataset
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tadataka.vo.semi_dense.frame import Frame
from tadataka.vo.semi_dense import validity
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.vo.semi_dense.propagation import (
    propagate, detect_intensity_change, warp_image)
from tadataka.vo.semi_dense.semi_dense import InverseDepthMapEstimator
from tadataka.vo.semi_dense.age import increment_age
from tadataka.vo.semi_dense.regularization import regularize
from tadataka.camera import CameraModel
from examples.plot import plot_depth


def plot_reprojection(keyframe, refframe):
    from matplotlib import pyplot as plt
    px, py = [200, 400]
    p = keyframe.camera_model.normalize(np.array([px, py], dtype=np.float64))
    q = warp2d((keyframe.pose.R, keyframe.pose.t),
               (refframe.pose.R, refframe.pose.t),
               p, keyframe.depth_map[py, px])
    qx, qy = refframe.camera_model.unnormalize(q)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(keyframe.image)
    ax.scatter(px, py)

    ax = fig.add_subplot(122)
    ax.imshow(refframe.image)
    ax.scatter(qx, qy)

    plt.show()


def gray_frame(frame):
    camera_model = CameraModel(
        frame.camera_model.camera_parameters,
        distortion_model=None
    )
    return Frame(camera_model, rgb2gray(frame.image),
                 frame.pose.R, frame.pose.t)


class ReferenceFrameSelector(object):
    def __init__(self, frame0, inv_depth_map0):
        self.age_map = np.zeros(inv_depth_map0.shape, dtype=np.int64)
        self.inv_depth_map0 = inv_depth_map0
        self.frames = []
        self.frame0 = frame0

    def update(self, frame1, inv_depth_map1):
        self.age_map = increment_age(
            self.age_map, self.inv_depth_map0,
            self.frame0.camera_model, frame1.camera_model,
            (self.frame0.R, self.frame0.t), (frame1.R, frame1.t)
        )

        self.frames.append(self.frame0)

        self.inv_depth_map0 = inv_depth_map1
        self.frame0 = frame1

    def __call__(self, u):
        x, y = u
        age = self.age_map[y, x]
        if age == 0:
            return None
        return self.frames[-age]

def plot_age_map(age_map):
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(age_map, cmap="gray")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


dataset = TumRgbdDataset("datasets/rgbd_dataset_freiburg1_xyz",
                         which_freiburg=1)
color_frames = dataset[0:50:10]

# dataset = NewTsukubaDataset("datasets/NewTsukubaStereoDataset")

frame = gray_frame(color_frames[0])
inv_depth_map = np.random.uniform(0.8, 1.2, frame.image.shape)
variance_map = 0.5 * np.ones(frame.image.shape)
reference_selector = ReferenceFrameSelector(frame, inv_depth_map)
default_validity = 0.5
validity_map = default_validity * np.ones(frame.image.shape)

for i in range(1, len(color_frames)-1):
    frame0_ = color_frames[i+0]
    frame1_ = color_frames[i+1]
    frame0 = gray_frame(frame0_)
    frame1 = gray_frame(frame1_)

    estimator = InverseDepthMapEstimator(
        frame0, sigma_i=0.01, sigma_l=0.02,
        step_size_ref=0.01, min_gradient=20.0
    )

    reference_selector.update(frame0, inv_depth_map)

    inv_depth_map, variance_map, flag_map = estimator(
        reference_selector, inv_depth_map, variance_map
    )
    success = flag_map==FLAG.SUCCESS
    validity_map = validity.decrease(validity_map, success)
    validity_map = validity.increase(validity_map, ~success)

    inv_depth_map = regularize(inv_depth_map, variance_map)

    plot_depth(frame0.image, reference_selector.age_map,
               flag_map, validity_map, frame0_.depth_map,
               invert_depth(inv_depth_map), variance_map)

    pose10 = frame1_.pose.inv() * frame0_.pose

    warp10 = Warp2D(frame0.camera_model, frame0.camera_model,
                    pose10, WorldPose.identity())

    mask = detect_intensity_change(warp10, frame0.image, frame1.image,
                                   inv_depth_map)
    validity_map = validity.increase(validity_map, mask)
    validity_map = warp_image(warp10, invert_depth(inv_depth_map),
                              validity_map, default_value=default_validity)

    # TODO collision handling
    inv_depth_map, variance_map = propagate(
        frame0.camera_model, frame1.camera_model,
        warp10, inv_depth_map, variance_map
    )

    plot_depth(frame0.image, reference_selector.age_map,
               flag_map, validity_map, frame0_.depth_map,
               invert_depth(inv_depth_map), variance_map)
