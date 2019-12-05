import warnings

import numpy as np
from tadataka import _triangulation as TR
from tadataka.pose import Pose
from tadataka.exceptions import InvalidDepthException
from tadataka.features import empty_match


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

    return point, TR.depths_are_valid(depth0, depth1, min_depth)


class Triangulation(object):
    def __init__(self, pose0, pose1, keypoints0, keypoints1):
        self.pose0, self.pose1 = pose0, pose1
        self.keypoints0, self.keypoints1 = keypoints0, keypoints1

    def triangulate_(self, index0, index1):
        keypoint0 = self.keypoints0[index0]
        keypoint1 = self.keypoints1[index1]
        return linear_triangulation(self.pose0, self.pose1, keypoint0, keypoint1)

    def triangulate(self, matches01):
        points = []
        depth_mask = []
        for index0, index1 in matches01:
            point, depth_is_positive = self.triangulate_(index0, index1)
            points.append(point)
            depth_mask.append(depth_is_positive)
        return np.array(points), np.array(depth_mask)


def triangulate(pose0, pose1, keypoints0, keypoints1, matches01):
    t = Triangulation(pose0, pose1, keypoints0, keypoints1)
    point_array, depth_mask = t.triangulate(matches01)
    # preserve points that have positive depths
    return point_array[depth_mask], matches01[depth_mask]
