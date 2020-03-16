import warnings

import numpy as np
from tadataka import _triangulation as TR
from tadataka.exceptions import InvalidDepthException
from tadataka.feature import empty_match
from tadataka.matrix import to_homogeneous
from tadataka.pose import LocalPose


class TwoViewTriangulation(object):
    def __init__(self, pose0, pose1):
        """
        pose0, pose1: Poses in the local coordinate system
        """
        self.triangulator = Triangulation([pose0, pose1])

    def triangulate(self, keypoints0, keypoints1):
        assert(keypoints0.shape == keypoints1.shape)
        keypoints = np.stack((keypoints0, keypoints1))
        return self.triangulator.triangulate(keypoints)


class Triangulation(object):
    def __init__(self, poses):
        """
        poses: List of poses in the local coordinate system
        """
        self.rotations = np.array([pose.rotation.as_matrix() for pose in poses])
        self.translations = np.array([pose.t for pose in poses])

    def triangulate(self, keypoints):
        return TR.linear_triangulation(self.rotations, self.translations,
                                       keypoints)


class DepthsFromTriangulation(object):
    def __init__(self, pose0, pose1):
        assert(isinstance(pose0, LocalPose))
        assert(isinstance(pose1, LocalPose))
        self.R0, self.t0 = pose0.R, pose0.t
        self.R1, self.t1 = pose1.R, pose1.t

    def __call__(self, keypoint0, keypoint1):
        """
        pose0, pose1 : Poses in the local coordinate system
        """

        def to_homogeneous(x):
            return np.append(x, 1)

        # y0 = inv(K) * homogeneous(x0)
        # y1 = inv(K) * homogeneous(x1)
        # In this implementation, we assume K = I

        # R0 * X + t0 = depth0 * y0
        # R1 * X + t1 = depth1 * y1

        # X = R0.T * (depth0 * y0 - t0) = depth0 * R0.T * y0 - R0.T * t0
        # X = R1.T * (depth1 * y1 - t1) = depth1 * R1.T * y1 - R1.T * t1
        # 0 = (depth0 * R0.T * y0 - R0.T * t0) - (depth1 * R1.T * y1 - R1.T * t1)
        # R0.T * t0 - R1.T * t1 = depth0 * R0.T * y0 - depth1 * R1.T * y1
        #                       = dot([R0.T * y0, -R1.T * y1], [depth0, depth1])

        R0, t0 = self.R0, self.t0
        R1, t1 = self.R1, self.t1

        y0 = to_homogeneous(keypoint0)
        y1 = to_homogeneous(keypoint1)
        A = np.column_stack((np.dot(R0.T, y0), -np.dot(R1.T, y1)))
        b = np.dot(R0.T, t0) - np.dot(R1.T, t1)
        depths, residuals, rank, s = np.linalg.lstsq(A, b)
        return depths


def depth_from_triangulation(R, t, keypoint0, keypoint1):
    """
    Estimate the depth corresponding to the keypoint in the first view

    Args:
        R, t: Pose change from viewpoint 0 to viewpoint 1
        keypoint0 (np.ndarray): Keypoint on the normalized image plane
            in the 0th viewpoint
        keypoint1 (np.ndarray): Keypoint on the normalized image plane
            in the 1st viewpoint
    Returns:
        Depth corresponding to the first keypoint
    """
    index = np.argmax(t[0:2])
    x0 = to_homogeneous(keypoint0)
    n = t[index] - t[2] * keypoint1[index]
    d = np.dot(R[2], x0) * keypoint1[index] - np.dot(R[index], x0)
    return n / d
