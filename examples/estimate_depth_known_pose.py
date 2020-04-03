import numpy as np
from tadataka.pose import WorldPose
from tadataka.warp import warp2d, Warp2D
from tadataka.dataset import NewTsukubaDataset, TumRgbdDataset
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.vo.semi_dense.frame import Frame
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.vo.semi_dense.propagation import (
    propagate, detect_intensity_change)
from tadataka.vo.semi_dense.semi_dense import InverseDepthMapEstimator
from tadataka.vo.semi_dense.regularization import regularize
from tadataka.vo.semi_dense.frame_selection import ReferenceFrameSelector
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


def update(keyframe, ref_selector, prior_inv_depth_map, prior_variance_map):
    estimator = InverseDepthMapEstimator(
        keyframe,
        sigma_i=0.01, sigma_l=0.02,
        step_size_ref=0.01, min_gradient=20.0
    )

    inv_depth_map, variance_map, flag_map = estimator(
        ref_selector, prior_inv_depth_map, prior_variance_map
    )

    inv_depth_map, variance_map = fusion(prior_inv_depth_map, inv_depth_map,
                                         prior_variance_map, variance_map)

    return inv_depth_map, variance_map, flag_map


dataset = TumRgbdDataset("datasets/rgbd_dataset_freiburg1_xyz",
                         which_freiburg=1)
color_frames = dataset[0:50:10]

# dataset = NewTsukubaDataset("datasets/NewTsukubaStereoDataset")

frame = Frame(color_frames[0])
inv_depth_map = np.random.uniform(0.8, 1.2, frame.image.shape)
variance_map = 0.5 * np.ones(frame.image.shape)
ref_selector = ReferenceFrameSelector(frame, inv_depth_map)

for i in range(1, len(color_frames)-1):
    frame0_ = color_frames[i+0]
    frame1_ = color_frames[i+1]
    frame0 = Frame(frame0_)
    frame1 = Frame(frame1_)

    ref_selector.update(frame0, inv_depth_map)

    inv_depth_map, variance_map, flag_map = update(
        frame0, ref_selector, inv_depth_map, variance_map
    )

    plot_depth(frame0.image, ref_selector.age_map,
               flag_map, frame0_.depth_map,
               invert_depth(inv_depth_map), variance_map)

    inv_depth_map = regularize(inv_depth_map, variance_map)

    plot_depth(frame0.image, ref_selector.age_map,
               flag_map, frame0_.depth_map,
               invert_depth(inv_depth_map), variance_map)

    pose10 = frame1_.pose.inv() * frame0_.pose

    warp10 = Warp2D(frame0.camera_model, frame0.camera_model,
                    pose10, WorldPose.identity())

    inv_depth_map, variance_map = propagate(warp10,
                                            inv_depth_map, variance_map)

    plot_depth(frame0.image, ref_selector.age_map,
               flag_map, frame0_.depth_map,
               invert_depth(inv_depth_map), variance_map)
