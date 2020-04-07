import numpy as np
from skimage.color import rgb2gray
from tadataka.pose import WorldPose
from tadataka.camera import CameraModel
from tadataka.warp import warp2d, Warp2D
from tadataka.dataset import NewTsukubaDataset, TumRgbdDataset
from tadataka.vo.semi_dense.age import increment_age
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.vo.semi_dense.frame import Frame
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.vo.semi_dense.propagation import (
    propagate, detect_intensity_change)
from tadataka.vo.semi_dense.semi_dense import InverseDepthMapEstimator
from tadataka.vo.semi_dense.regularization import regularize
from tadataka.vo.semi_dense.frame_selection import ReferenceSelector
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
        sigma_i=0.1, sigma_l=0.2,
        step_size_ref=0.001, min_gradient=20.0
    )

    inv_depth_map, variance_map, flag_map = estimator(
        ref_selector, prior_inv_depth_map, prior_variance_map
    )

    inv_depth_map, variance_map = fusion(prior_inv_depth_map, inv_depth_map,
                                         prior_variance_map, variance_map)

    inv_depth_map = regularize(inv_depth_map, variance_map)

    return inv_depth_map, variance_map, flag_map


def to_perspective(camera_model):
    return CameraModel(
        camera_model.camera_parameters,
        distortion_model=None
    )


def create_frame(camera_model, image, pose):
    return Frame(to_perspective(camera_model), rgb2gray(image), pose)


def plot_refframes(refframes, keyframe, age_map):
    from matplotlib import pyplot as plt
    N = len(refframes) + 2

    fig = plt.figure()
    for i, frame in enumerate(refframes):
        ax = fig.add_subplot(1, N, i+1)
        ax.set_title(f"refframe {i}")
        ax.imshow(frame.image, cmap="gray")

    ax = fig.add_subplot(1, N, len(refframes)+1)
    ax.set_title("keyframe")
    ax.imshow(keyframe.image, cmap="gray")

    ax = fig.add_subplot(1, N, len(refframes)+2)
    ax.set_title("pixel age")
    ax.imshow(age_map, cmap="gray")

    plt.show()


def main():
    dataset = TumRgbdDataset("datasets/rgbd_dataset_freiburg1_desk",
                             which_freiburg=1)
    color_frames = dataset[100:130]

    frame0_ = color_frames[0]
    frame0 = create_frame(frame0_.camera_model, frame0_.image, frame0_.pose)

    age_map0 = np.zeros(frame0.image.shape, dtype=np.int64)
    # inv_depth_map0 = invert_depth(frame0_.depth_map)
    inv_depth_map0 = np.random.uniform(0.1, 10.0, frame0.image.shape)
    variance_map0 = 10.0 * np.ones(frame0.image.shape)

    refframes = []

    for i in range(1, len(color_frames)):
        frame1_ = color_frames[i]
        frame1 = create_frame(frame1_.camera_model, frame1_.image, frame1_.pose)

        warp10 = Warp2D(frame0.camera_model, frame0.camera_model,
                        frame0_.pose, frame1_.pose)

        prior_inv_depth_map1, prior_variance_map1 = propagate(
            warp10, inv_depth_map0, variance_map0
        )
        age_map1 = increment_age(age_map0, warp10, inv_depth_map0)

        refframes.append(frame0)
        # plot_refframes(refframes, frame1, age_map1)

        inv_depth_map1, variance_map1, flag_map1 = update(
            frame1, ReferenceSelector(refframes, age_map1),
            prior_inv_depth_map1, prior_variance_map1
        )

        plot_depth(frame1.image, age_map1,
                   flag_map1, frame1_.depth_map,
                   invert_depth(inv_depth_map1), variance_map1)

        frame0 = frame1
        age_map0 = age_map1
        inv_depth_map0 = inv_depth_map1
        variance_map0 = variance_map1

main()
