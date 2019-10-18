import warnings

from autograd import numpy as np
from vitamine import _triangulation as TR
from vitamine.pose import Pose
from vitamine.exceptions import InvalidDepthException


def depth_mask_condition(mask, min_positive_dpth_ratio=0.8):
    return np.sum(mask) / len(mask) >= min_positive_dpth_ratio


def pose_point_from_keypoints(keypoints0, keypoints1, matches01):
    R, t, points, valid_depth_mask = TR.pose_point_from_keypoints(
        keypoints0[matches01[:, 0]],
        keypoints1[matches01[:, 1]]
    )

    if not depth_mask_condition(valid_depth_mask):
        warnings.warn(
            "Most of points are behind cameras. Maybe wrong matches?"
        )

    return (Pose.identity(), Pose(R, t),
            points[valid_depth_mask], matches01[valid_depth_mask])


def points_from_known_poses(pose0, pose1, keypoints0, keypoints1, matches01):
    points, valid_depth_mask = TR.points_from_known_poses(
        pose0.R, pose1.R, pose0.t, pose1.t,
        keypoints0[matches01[:, 0]],
        keypoints1[matches01[:, 1]]
    )

    if not depth_mask_condition(valid_depth_mask):
        warnings.warn(
            "Most of points are behind cameras. Maybe wrong matches?"
        )

    return points[valid_depth_mask], matches01[valid_depth_mask]


def linear_triangulation(pose0, pose1, keypoint0, keypoint1, min_depth=0.0):
    point, depth0, depth1 = TR.linear_triangulation(
        pose0.R, pose1.R, pose0.t, pose1.t,
        keypoint0, keypoint1
    )

    if not TR.depths_are_valid(depth0, depth1, min_depth):
        raise InvalidDepthException(
            "Triangulated point has insufficient depth"
        )

    return point
