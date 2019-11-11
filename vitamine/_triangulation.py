from autograd import numpy as np

from vitamine.depth import depth_condition, warn_points_behind_cameras
from vitamine.matrix import solve_linear, motion_matrix
from vitamine.so3 import rodrigues

# Equation numbers are the ones in Multiple View Geometry


def calc_depth(P, x):
    return np.dot(P[2], x)


def linear_triangulation(R0, R1, t0, t1, keypoints0, keypoints1):
    P0 = motion_matrix(R0, t0)
    P1 = motion_matrix(R1, t1)

    x0, y0 = keypoints0
    x1, y1 = keypoints1

    # See section 12.2 for details
    A = np.vstack([
        x0 * P0[2] - P0[0],
        y0 * P0[2] - P0[1],
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1],
    ])
    x = solve_linear(A)

    # normalize so that x / x[3] be a homogeneous vector [x y z 1]
    # and extract the first 3 elements
    # assert(x[3] != 0)
    if np.isclose(x[3], 0):
        return np.inf * np.ones(3), np.nan, np.nan

    x = x / x[3]
    # calculate depths for utilities
    return x[0:3], calc_depth(P0, x), calc_depth(P1, x)


def depths_are_valid(depth0, depth1, min_depth):
    return depth0 > min_depth and depth1 > min_depth


def triangulation_(R0, R1, t0, t1, keypoints0, keypoints1, min_depth=0.0):
    """
    Reconstruct 3D points from 2 camera poses.
    The first camera pose is assumed to be R = identity, t = zeros.
    """

    n_points = keypoints0.shape[0]

    points = np.empty((n_points, 3))

    depth_mask = np.zeros(n_points, dtype=np.bool)

    for i in range(n_points):
        points[i], depth0, depth1 = linear_triangulation(
            R0, R1, t0, t1, keypoints0[i], keypoints1[i])
        depth_mask[i] = depths_are_valid(depth0, depth1, min_depth)
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
