import numpy as np
from skimage.color import rgb2gray, gray2rgb

from tqdm import tqdm
from tadataka.camera import CameraModel
from tadataka.coordinates import image_coordinates
from tadataka.dataset import TumRgbdDataset
from tadataka.vo.semi_dense.frame import Frame
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.pose import WorldPose
from tadataka.vo.dvo import PoseChangeEstimator
from tadataka.matrix import to_homogeneous
from tadataka.rigid_transform import transform
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.vo.semi_dense.propagation import propagate
from tadataka.dataset import NewTsukubaDataset
from tests.dataset.path import new_tsukuba

from examples.plot import plot_depth, plot_prior
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tadataka.vo.semi_dense.semi_dense import InverseDepthMapEstimator


def gray_frame(frame):
    return Frame(frame.camera_model, rgb2gray(frame.image), frame.pose)


def update(keyframe, refframe, prior_inv_depth_map, prior_variance_map):
    estimator = InverseDepthMapEstimator(
        keyframe,
        sigma_i=0.01, sigma_l=0.02,
        step_size_ref=0.01, min_gradient=20.0
    )

    inv_depth_map, variance_map, flag_map = estimator(
        refframe, prior_inv_depth_map, prior_variance_map
    )

    inv_depth_map, variance_map = fusion(prior_inv_depth_map, inv_depth_map,
                                         prior_variance_map, variance_map)

    return inv_depth_map, variance_map, flag_map


# dataset = TumRgbdDataset("datasets/rgbd_dataset_freiburg1_xyz",
#                          which_freiburg=1)

dataset = NewTsukubaDataset("datasets/NewTsukubaStereoDataset")
frames = dataset[:200]

inv_depth_map = invert_depth(frames[0][0].depth_map)
variance_map = 10.0 * np.ones(frames[0][0].depth_map.shape)

EPSILON = 1e-16

trajectory_true = []
trajectory_pred = []
pose_pred = WorldPose.identity()
pose_true = WorldPose.identity()
offset = 4
for i in range(len(frames)-offset):
    kf0_ = frames[i][0]
    kf1_ = frames[i+1][0]
    keyframe0 = gray_frame(kf0_)
    keyframe1 = gray_frame(kf1_)
    refframe = gray_frame(frames[i+offset][0])

    # plot_prior(keyframe0.image, kf0_.depth_map,
    #            invert_depth(inv_depth_map), variance_map)

    inv_depth_map, variance_map, flag_map = update(
        keyframe0, refframe, inv_depth_map, variance_map
    )

    # plot_depth(keyframe0.image, refframe.image, flag_map,
    #            kf0_.depth_map, invert_depth(inv_depth_map), variance_map)

    estimate_pose = PoseChangeEstimator(
        keyframe0.camera_model, keyframe1.camera_model
    )

    depth_map0 = invert_depth(inv_depth_map)
    pose10_pred = estimate_pose(keyframe0.image, depth_map0,
                                keyframe1.image,
                                weight_map=1/(variance_map+EPSILON))

    pose_pred = pose10_pred * pose_pred

    inv_depth_map, variance_map = propagate(keyframe0.camera_model,
                                            keyframe1.camera_model,
                                            pose10_pred, inv_depth_map,
                                            variance_map)

    pose10_true = keyframe1.pose.inv() * keyframe0.pose
    pose_true = pose10_true * pose_true

    print("pred", pose10_pred)
    print("true", pose10_true)

    trajectory_pred.append(pose_pred.t)
    trajectory_true.append(pose_true.t)

trajectory_true = np.array(trajectory_true)
trajectory_pred = np.array(trajectory_pred)
print(trajectory_true)
print(trajectory_pred)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(trajectory_pred[:, 0], trajectory_pred[:, 1], trajectory_pred[:, 2],
        label="pred")
ax.plot(trajectory_true[:, 0], trajectory_true[:, 1], trajectory_true[:, 2],
        label="true")
plt.legend()
plt.show()
