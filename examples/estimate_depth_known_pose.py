import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from tqdm import tqdm

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
        step_size_ref=0.01, min_gradient=20.0
    )

    inv_depth_map, variance_map, flag_map = estimator(
        ref_selector, prior_inv_depth_map, prior_variance_map
    )

    inv_depth_map, variance_map = fusion(prior_inv_depth_map, inv_depth_map,
                                         prior_variance_map, variance_map)

    inv_depth_map = regularize(inv_depth_map, variance_map)

    return inv_depth_map, variance_map, flag_map


def to_perspective(camera_model):
    return CameraModel(camera_model.camera_parameters,
                       distortion_model=None)


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


def get(frame, scale=1.0):
    camera_model = to_perspective(frame.camera_model)
    camera_model = camera.resize(camera_model, scale)

    image = rgb2gray(frame.image)
    shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))
    image = resize(image, shape)
    return camera_model, image


def dvo(camera_model0, camera_model1, image0, image1, depth_map, weights):
    estimator = PoseChangeEstimator(camera_model0, camera_model1,
                                    n_coarse_to_fine=7)
    return estimator(image0, depth_map, image1, weights)


def main():
    scale = 1.0
    dataset = NewTsukubaDataset("datasets/NewTsukubaStereoDataset",
                                condition="fluorescent")
    color_frames = [fl for fl, fr in dataset[0:1000]]
    # dataset = TumRgbdDataset("datasets/rgbd_dataset_freiburg1_desk",
    #                          which_freiburg=1)
    # color_frames = dataset[:30]

    # shape_ = get(color_frames[0])[1].shape[0:2]
    # pose0 = WorldPose.identity()
    # age_map0 = np.zeros(shape_, dtype=np.int64)
    # inv_depth_map0 = np.random.uniform(0.1, 10.0, shape_)
    # variance_map0 = np.ones(shape_)

    trajectory_true = []
    trajectory_pred = []
    # refframes = []

    pose_pred = WorldPose.identity()
    pose_true = WorldPose.identity()
    for i in tqdm(range(len(color_frames)-1)):
        frame0_, frame1_ = color_frames[i+0], color_frames[i+1]

        camera_model0, image0 = get(frame0_, scale)
        camera_model1, image1 = get(frame1_, scale)

        depth_map0 = resize(frame0_.depth_map, image1.shape)
        # # if i == 0:
        # #     pose10 = frame1_.pose.inv() * frame0_.pose
        # # else:
        pose10_true = frame1_.pose.inv() * frame0_.pose
        pose10_pred = dvo(camera_model0, camera_model1, image0, image1,
                          depth_map0, None)
        error_true = photometric_error(
            LocalWarp2D(camera_model0, camera_model1, pose10_true),
                        image0, depth_map0, image1)
        error_pred = photometric_error(
            LocalWarp2D(camera_model0, camera_model1, pose10_pred),
                        image0, depth_map0, image1)
        print("i =", i)
        print("pose10_true :", pose10_true)
        print("pose10_pred :", pose10_pred)
        # print("diff       ", pose10_true * pose10_pred.inv())

        print("error true = {:.6f}".format(error_true))
        print("error pred = {:.6f}".format(error_pred))

        if False:  # error_pred > error_true:
            from examples.plot import plot_warp
            plot_warp(LocalWarp2D(camera_model0, camera_model1, pose10_true),
                      image0, depth_map0, image1)
            plot_warp(LocalWarp2D(camera_model0, camera_model1, pose10_pred),
                      image0, depth_map0, image1)
        pose_pred = pose_pred * pose10_pred.inv()
        pose_true = pose_true * pose10_true.inv()
        trajectory_pred.append(pose_pred.t)
        trajectory_true.append(pose_true.t)
        continue

        # pose1 = pose10 * pose0

        # warp10 = Warp2D(camera_model0, camera_model1, pose0, pose1)
        # age_map1 = increment_age(age_map0, warp10, inv_depth_map0)

        # refframes.append(Frame(camera_model0, image0, pose0))
        # # plot_refframes(refframes, frame1, age_map1)

        # inv_depth_map1, variance_map1, flag_map1 = update(
        #     Frame(camera_model1, image1, pose1),
        #     ReferenceSelector(refframes, age_map1),
        #     *propagate(warp10, inv_depth_map0, variance_map0)
        # )

        # # plot_depth(image1, age_map1, flag_map1,
        # #            resize(frame1_.depth_map, image1.shape),
        # #            invert_depth(inv_depth_map1), variance_map1)

        # age_map0 = age_map1
        # inv_depth_map0 = inv_depth_map1
        # variance_map0 = variance_map1
        # pose0 = pose1

        # trajectory_pred.append(pose1.t)
        # trajectory_true.append(frame1_.pose.t)


    plot_trajectory(np.array(trajectory_true), np.array(trajectory_pred))

main()
