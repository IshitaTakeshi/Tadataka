import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from matplotlib import pyplot as plt

from tadataka import camera
from tadataka.pose import WorldPose
from tadataka.camera import CameraModel, CameraParameters
from tadataka.warp import warp2d, Warp2D
from tadataka.dataset import NewTsukubaDataset, TumRgbdDataset
from tadataka.vo.dvo import PoseChangeEstimator
from tadataka.vo.semi_dense.age import increment_age
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.vo.semi_dense.frame import Frame
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.metric import photometric_error
from tadataka.warp import LocalWarp2D
from tadataka.vo.semi_dense.propagation import (
    propagate, detect_intensity_change)
from tadataka.vo.semi_dense.semi_dense import InverseDepthMapEstimator
from tadataka.vo.semi_dense.regularization import regularize
from tadataka.vo.semi_dense.frame_selection import ReferenceSelector
from examples.plot import plot_depth, plot_trajectory


def plot_reprojection(keyframe, refframe):
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


def plot_refframes(refframes, keyframe, age_map):
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


def get(frame):
    return to_perspective(frame.camera_model), rgb2gray(frame.image)


def dvo_(cm0, cm1, I0, D0, I1, W, scale=1/6):
    estimator = PoseChangeEstimator(camera.resize(cm0, scale),
                                    camera.resize(cm1, scale),
                                    n_coarse_to_fine=3)

    shape = (int(I0.shape[0] * scale), int(I0.shape[1] * scale))
    return estimator(resize(I0, shape), resize(D0, shape),
                     resize(I1, shape), resize(W, shape))


def dvo(camera_model0, camera_model1, image0, image1,
        inv_depth_map, variance_map):
    return dvo_(camera_model0, camera_model1,
                image0, invert_depth(inv_depth_map),
                image1, invert_depth(variance_map))


def main():
    dataset = TumRgbdDataset("datasets/rgbd_dataset_freiburg1_desk",
                             which_freiburg=1)
    color_frames = dataset[100:200]

    pose0 = WorldPose.identity()
    age_map0 = np.zeros(color_frames[0].image.shape[0:2], dtype=np.int64)
    # inv_depth_map0 = np.random.uniform(0.1, 10.0, image0.shape)
    variance_map0 = np.ones(color_frames[0].image.shape[0:2])

    trajectory_true = []
    trajectory_pred = []
    refframes = []
    for i in range(len(color_frames)-1):
        frame0_, frame1_ = color_frames[i+0], color_frames[i+1]

        camera_model0, image0 = get(frame0_)
        camera_model1, image1 = get(frame1_)

        inv_depth_map0 = invert_depth(frame0_.depth_map)

        pose10 = dvo(camera_model0, camera_model1, image0, image1,
                     inv_depth_map0, np.ones(image0.shape))
        pose1 = pose10 * pose0
        print("pose10 pred", pose10)
        print("pose10 true", frame1_.pose * frame0_.pose.inv())

        warp10 = Warp2D(camera_model0, camera_model1, pose0, pose1)
        age_map1 = increment_age(age_map0, warp10, inv_depth_map0)

        refframes.append(Frame(camera_model0, image0, pose0))
        # plot_refframes(refframes, frame1, age_map1)

        inv_depth_map1, variance_map1, flag_map1 = update(
            Frame(camera_model1, image1, pose1),
            ReferenceSelector(refframes, age_map1),
            *propagate(warp10, inv_depth_map0, variance_map0)
        )

        # plot_depth(image1, age_map1, flag_map1, frame1_.depth_map,
        #            invert_depth(inv_depth_map1), variance_map1)

        age_map0 = age_map1
        inv_depth_map0 = inv_depth_map1
        variance_map0 = variance_map1
        pose0 = pose1

        trajectory_pred.append(pose1.t)
        trajectory_true.append(frame1_.pose.t)

    plot_trajectory(np.array(trajectory_true), np.array(trajectory_pred))


main()
