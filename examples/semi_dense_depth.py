import numpy as np
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize

from tqdm import tqdm
from tadataka.camera import CameraModel
from tadataka.rigid_motion import LeastSquaresRigidMotion
from tadataka.coordinates import image_coordinates
from tadataka.dataset import TumRgbdDataset
from tadataka.vo.semi_dense.frame import Frame
from tadataka.vo.semi_dense.frame_selection import ReferenceFrameSelector
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.vo.semi_dense.propagation import propagate
from tadataka.pose import WorldPose
from tadataka.vo.dvo import PoseChangeEstimator
from tadataka.matrix import to_homogeneous
from tadataka.warp import Warp2D
from tadataka.rigid_transform import transform, Transform
from tadataka.dataset import NewTsukubaDataset
from tests.dataset.path import new_tsukuba

from examples.plot import plot_depth, plot_prior
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tadataka.vo.semi_dense.semi_dense import InverseDepthMapEstimator


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


def resize_camera_model(cm, ratio):
    from tadataka.camera import CameraParameters
    return CameraModel(
        CameraParameters(cm.camera_parameters.focal_length * ratio,
                         cm.camera_parameters.offset * ratio),
        cm.distortion_model
    )


def estimate_pose_change_(cm0, cm1, I0, D0, I1, W, ratio=1/6):
    estimator = PoseChangeEstimator(resize_camera_model(cm0, ratio),
                                    resize_camera_model(cm1, ratio))

    shape = (int(I0.shape[0] * ratio), int(I0.shape[1] * ratio))
    return estimator(resize(I0, shape), resize(D0, shape),
                     resize(I1, shape), resize(W, shape))


def estimate_pose_change(frame0, frame1, inv_depth_map, variance_map):
    return estimate_pose_change_(frame0.camera_model, frame1.camera_model,
                                 frame0.image, invert_depth(inv_depth_map),
                                 frame1.image, invert_depth(variance_map))


def main():
    dataset = TumRgbdDataset("datasets/rgbd_dataset_freiburg1_desk",
                             which_freiburg=1)

    frames = dataset[200:220]

    inv_depth_map = invert_depth(frames[0].depth_map)
    variance_map = 10.0 * np.ones(frames[0].depth_map.shape)
    ref_selector = ReferenceFrameSelector(Frame(frames[0]), inv_depth_map)

    trajectory_true = []
    trajectory_pred = []
    pose_pred = WorldPose.identity()
    pose_true = WorldPose.identity()
    for i in range(len(frames)-1):
        frame0_ = frames[i+0]
        frame1_ = frames[i+1]
        frame0 = Frame(frame0_)
        frame1 = Frame(frame1_)

        inv_depth_map, variance_map, flag_map = update(
            frame0, ref_selector, inv_depth_map, variance_map
        )

        pose10_pred = estimate_pose_change(frame0, frame1,
                                           inv_depth_map, variance_map)

        pose_pred = pose10_pred * pose_pred

        warp10 = Warp2D(frame0.camera_model, frame0.camera_model,
                        pose10_pred, WorldPose.identity())
        inv_depth_map, variance_map = propagate(warp10,
                                                inv_depth_map, variance_map)

        pose10_true = frame1_.pose.inv() * frame0_.pose
        pose_true = pose10_true * pose_true

        trajectory_pred.append(pose_pred.t)
        trajectory_true.append(pose_true.t)

    trajectory_true = np.array(trajectory_true)
    trajectory_pred = np.array(trajectory_pred)
    R, t, s = LeastSquaresRigidMotion(trajectory_pred, trajectory_true).solve()
    trajectory_pred = Transform(R, t, s)(trajectory_pred)
    print("MSE: ", np.power(trajectory_pred - trajectory_true, 2).mean())

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot(trajectory_pred[:, 0], trajectory_pred[:, 1], trajectory_pred[:, 2],
    #         label="pred")
    # ax.plot(trajectory_true[:, 0], trajectory_true[:, 1], trajectory_true[:, 2],
    #         label="true")
    # plt.legend()
    # plt.show()

main()
