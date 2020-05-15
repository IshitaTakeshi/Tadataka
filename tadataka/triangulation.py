import warnings

import numpy as np
from tadataka.exceptions import InvalidDepthException
from tadataka.feature import empty_match
from tadataka.matrix import to_homogeneous, solve_linear
from tadataka._triangulation import calc_depth0_


def linear_triangulation_(rotations, translations, keypoints):
    # keypoints.shape == (n_poses, 2)
    # rotations.shape == (n_poses, 3, 3)
    # translations.shape == (n_poses, 3)
    assert(rotations.shape[0] == translations.shape[0] == keypoints.shape[0])

    def calc_depths(x):
        return np.dot(rotations[:, 2], x) + translations[:, 2]

    # let
    # A = np.vstack([
    #     x0 * P0[2] - P0[0],
    #     y0 * P0[2] - P0[1],
    #     x1 * P1[2] - P1[0],
    #     y1 * P1[2] - P1[1],
    #     x2 * P2[2] - P2[0],
    #     y2 * P2[2] - P2[1],
    #     ...
    # ])
    # and compute argmin_x ||Ax||
    #
    # See Multiple View Geometry section 12.2 for details

    A = np.empty((2 * keypoints.shape[0], 4))

    A[0::2, 0:3] = keypoints[:, [0]] * rotations[:, 2] - rotations[:, 0]
    A[1::2, 0:3] = keypoints[:, [1]] * rotations[:, 2] - rotations[:, 1]

    A[0::2, 3] = keypoints[:, 0] * translations[:, 2] - translations[:, 0]
    A[1::2, 3] = keypoints[:, 1] * translations[:, 2] - translations[:, 1]

    x = solve_linear(A)

    if np.isclose(x[3], 0):
        return np.full(3, np.inf), np.full(keypoints.shape[0], np.nan)

    x = x[0:3] / x[3]

    return x, calc_depths(x)


def depths_are_valid(depths, min_depth):
    return np.all(depths > min_depth)


def linear_triangulation(rotations, translations, keypoints):
    """
    Args:
        rotations : np.ndarray (n_poses, 3, 3)
            rotation matrices of shape
        translations : np.ndarray (n_poses, 3)
            translation vectors of shape
        keypoints : np.ndarray (n_poses, n_keypoints, 2)
            keypoints observed in each viewpoint
    Returns:
        points : np.ndarray (n_keypoints, 3)
            Triangulated points
        depths : np.ndarray (n_keypoints, n_poses)
            Point depths from each viewpoint
    """

    # estimate a 3D point coordinate from multiple projections
    assert(rotations.shape[0] == translations.shape[0] == keypoints.shape[0])
    assert(rotations.shape[1:3] == (3, 3))
    assert(translations.shape[1] == 3)
    assert(keypoints.shape[2] == 2)

    n_poses, n_points = keypoints.shape[0:2]
    points = np.empty((n_points, 3))
    depths = np.empty((n_poses, n_points))
    for i in range(n_points):
        points[i], depths[:, i] = linear_triangulation_(
            rotations, translations, keypoints[:, i]
        )
    return points, depths


class TwoViewTriangulation(object):
    def __init__(self, pose0w, pose1w):
        self.triangulator = Triangulation([pose0w, pose1w])

    def triangulate(self, keypoints0: np.ndarray, keypoints1: np.ndarray):
        """
        Args:
            keypoints0: Normalized keypoints observed from viewpoint 0
            keypoints1: Normalized keypoints observed from viewpoint 1
        Returns:
            points: shape (points, depths)
                3D points in the world coordinate
            depths: shape (n_viewpoints, n_points)
                Point depths
        """

        assert(keypoints0.shape == keypoints1.shape)
        keypoints = np.stack((keypoints0, keypoints1))
        return self.triangulator.triangulate(keypoints)


class Triangulation(object):
    def __init__(self, poses):
        """
        poses: List of poses in the local coordinate system
        """
        self.rotations = np.array([pose.R for pose in poses])
        self.translations = np.array([pose.t for pose in poses])

    def triangulate(self, keypoints):
        """
        keypoints: np.ndarray (n_poses, n_keypoints, 2)
            keypoints observed in each viewpoint
        """
        return linear_triangulation(self.rotations, self.translations,
                                    keypoints)


class DepthsFromTriangulation(object):
    def __init__(self, pose0, pose1):
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


def calc_depth0(posew0, posew1, x0, x1):
    """
    Estimate the depth corresponding to x0

    Args:
        pose10 : Pose change from viewpoint 0 to viewpoint 1
        x0 (np.ndarray): Keypoint on the normalized image plane
            in the 0th viewpoint
        x1 (np.ndarray): Keypoint on the normalized image plane
            in the 1st viewpoint
    Returns:
        Depth corresponding to the first keypoint
    """
    # transformation from key camera coordinate to ref camera coordinate
    pose10 = posew1.inv() * posew0
    return calc_depth0_(pose10.R, pose10.t, x0, x1)
