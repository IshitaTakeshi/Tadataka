import numpy as np

from tadataka.depth import depth_condition, warn_points_behind_cameras
from tadataka.matrix import solve_linear, motion_matrix


def linear_triangulation(rotations, translations, keypoints):
    # estimate a 3D point coordinate from multiple projections
    assert(rotations.shape[0] == translations.shape[0] == keypoints.shape[0])
    assert(rotations.shape[1:3] == (3, 3))
    assert(translations.shape[1] == 3)
    assert(keypoints.shape[1] == 2)

    def calc_depths(x):
        return np.dot(rotations[:, 2], x) + translations[:, 2]

    # A = np.vstack([
    #     x0 * P0[2] - P0[0],
    #     y0 * P0[2] - P0[1],
    #     x1 * P1[2] - P1[0],
    #     y1 * P1[2] - P1[1],
    #     x2 * P2[2] - P2[0],
    #     y2 * P2[2] - P2[1],
    #     ...
    # ])
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


def two_view_triangulation(R0, R1, t0, t1, keypoint0, keypoint1):
    return linear_triangulation(
        np.array([R0, R1]),
        np.array([t0, t1]),
        np.array([keypoint0, keypoint1])
    )


def depths_are_valid(depths, min_depth):
    return np.all(depths > min_depth)


def triangulation_(R0, R1, t0, t1, keypoints0, keypoints1, min_depth=0.0):
    """
    Reconstruct 3D points from 2 camera poses.
    The first camera pose is assumed to be R = identity, t = zeros.
    """

    n_points = keypoints0.shape[0]

    points = np.empty((n_points, 3))

    depth_mask = np.zeros(n_points, dtype=np.bool)

    for i in range(n_points):
        points[i], depths = two_view_triangulation(
            R0, R1, t0, t1, keypoints0[i], keypoints1[i])
        depth_mask[i] = depths_are_valid(depths, min_depth)
    return points, depth_mask


def triangulation(R0, R1, t0, t1, keypoints0, keypoints1):
    assert(R0.shape == (3, 3))
    assert(R1.shape == (3, 3))
    assert(t0.shape == (3,))
    assert(t1.shape == (3,))
    assert(keypoints0.shape == keypoints1.shape)

    points, depth_mask = triangulation_(R0, R1, t0, t1, keypoints0, keypoints1)
    if not depth_condition(depth_mask):
        warn_points_behind_cameras()
    return points, depth_mask
