from autograd import numpy as np

from vitamine.exceptions import InvalidDepthsException
from vitamine import triangulation as TR
from vitamine.visual_odometry.pose import Pose


def depth_mask_condition(mask, min_positive_dpth_ratio=0.8):
    return np.sum(mask) / len(mask) >= min_positive_dpth_ratio


def pose_point_from_keypoints(keypoints0, keypoints1, matches01):
    R, t, points, valid_depth_mask = TR.pose_point_from_keypoints(
        keypoints0[matches01[:, 0]],
        keypoints1[matches01[:, 1]]
    )

    if not depth_mask_condition(valid_depth_mask):
        raise InvalidDepthsException(
            "Most of points are behind cameras. Maybe wrong matches?"
        )

    return Pose(R, t), points[valid_depth_mask], matches01[valid_depth_mask]


def points_from_known_poses(keypoints0, keypoints1, pose0, pose1, matches01):
    points, valid_depth_mask = TR.points_from_known_poses(
        pose0.R, pose1.R, pose0.t, pose1.t,
        keypoints0[matches01[:, 0]],
        keypoints1[matches01[:, 1]]
    )

    if not depth_mask_condition(valid_depth_mask):
        raise InvalidDepthsException(
            "Most of points are behind cameras. Maybe wrong matches?"
        )

    return points[valid_depth_mask], matches01[valid_depth_mask]
