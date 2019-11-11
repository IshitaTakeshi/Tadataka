import warnings

from autograd import numpy as np
from vitamine import _triangulation as TR
from vitamine.pose import Pose
from vitamine.exceptions import InvalidDepthException


def estimate_pose_change(keypoints0, keypoints1, matches01):
    R, t = TR.pose_point_from_keypoints(
        keypoints0[matches01[:, 0]],
        keypoints1[matches01[:, 1]]
    )

    return Pose(R, t)


def points_from_known_poses(pose0, pose1, keypoints0, keypoints1, matches01):
    points, depth_mask = TR.points_from_known_poses(
        pose0.R, pose1.R, pose0.t, pose1.t,
        keypoints0[matches01[:, 0]],
        keypoints1[matches01[:, 1]]
    )

    return points[depth_mask], matches01[depth_mask]


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
