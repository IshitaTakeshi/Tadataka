import warnings

from autograd import numpy as np
from vitamine import _triangulation as TR
from vitamine.pose import Pose
from vitamine.exceptions import InvalidDepthException


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


class Triangulation(object):
    def __init__(self, pose0, pose1, keypoints0, keypoints1):
        self.pose0, self.pose1 = pose0, pose1
        self.keypoints0, self.keypoints1 = keypoints0, keypoints1

    def triangulate(self, index0, index1):
        keypoint0 = self.keypoints0[index0]
        keypoint1 = self.keypoints1[index1]

        try:
            return linear_triangulation(self.pose0, self.pose1,
                                        keypoint0, keypoint1)
        except InvalidDepthException as e:
            print_error(e)
            return None
