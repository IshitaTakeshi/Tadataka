import itertools

import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from vitamine.matrix import solve_linear
from vitamine.rigid.rotation import tangent_so3
from vitamine.assertion import check_non_nan


# Equation numbers are the ones in Multiple View Geometry

Z = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 0]
])


W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])


def project(p, K):
    q = np.dot(K, p)
    return q / q[2]


def estimate_fundamental(keypoints0, keypoints1):
    # Eq. 11.3
    assert(keypoints0.shape == keypoints1.shape)

    N = keypoints0.shape[0]
    assert(N >= 8)

    XA, YA = keypoints0[:, 0], keypoints0[:, 1]
    XB, YB = keypoints1[:, 0], keypoints1[:, 1]
    A = np.vstack((XB * XA, XB * YA, XB,
                   YB * XA, YB * YA, YB,
                   XA, YA, np.ones(N))).T
    f = solve_linear(A)
    F = f.reshape(3, 3)
    return F


def fundamental_to_essential(F, K0, K1=None):
    if K1 is None:
        K1 = K0
    return K1.T.dot(F).dot(K0)


# TODO compute multiple points
def linear_triangulation(R0, R1, t0, t1, keypoints0, keypoints1, K):
    def calc_depth(P, x):
        return np.dot(P[2], x)

    def motion_matrix(R, t):
        T = np.empty((3, 4))
        T[0:3, 0:3] = R
        T[0:3, 3] = t
        return T

    P0 = np.dot(K, motion_matrix(R0, t0))
    P1 = np.dot(K, motion_matrix(R1, t1))

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
    assert(x[3] != 0)
    x = x / x[3]
    # calculate depths for utilities
    return x[0:3], calc_depth(P0, x), calc_depth(P1, x)


def projection_matrix(E, F, K):
    R, t = extract_pose(E)
    e = np.dot(K, t)  # project(t, K)
    S = tangent_so3(e.reshape(1, 3))[0]

    P = np.empty((3, 4))
    P[0:3, 0:3] = S.dot(F)
    P[0:3, 3] = e
    return P


def extract_poses(E):
    """
    Get rotation and translation from the essential matrix.
    There are 2 solutions and this functions returns both of them.
    """

    # Eq. 9.14
    U, _, VH = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U = -U

    if np.linalg.det(VH) < 0:
        VH = -VH

    R1 = U.dot(W).dot(VH)
    R2 = U.dot(W.T).dot(VH)

    S = -U.dot(W).dot(np.diag([1, 1, 0])).dot(U.T)
    t1 = np.array([S[2, 1], S[0, 2], S[1, 0]])
    t2 = -t1
    return R1, R2, t1, t2


def points_from_known_poses(R0, R1, t0, t1, keypoints0, keypoints1, K):
    """
    Reconstruct 3D points from 2 camera poses.
    The first camera pose is assumed to be R = identity, t = zeros.
    """

    assert(R0.shape == (3, 3))
    assert(R1.shape == (3, 3))
    assert(t0.shape == (3,))
    assert(t1.shape == (3,))
    assert(keypoints0.shape == keypoints1.shape)

    n_points = keypoints0.shape[0]

    points = np.empty((n_points, 3))
    depths_are_valid = True
    for i in range(n_points):
        points[i], depth0, depth1 = linear_triangulation(
            R0, R1, t0, t1, keypoints0[i], keypoints1[i], K)

        depth_is_valid = depth0 > 0 and depth1 > 0
        depths_are_valid = depths_are_valid and depth_is_valid
    return points, depths_are_valid


def points_from_unknown_poses(keypoints0, keypoints1, K):
    """
    keypoints[01].shape == (n_points, 2)
    """
    assert(keypoints0.shape == keypoints1.shape)

    R0, t0 = np.identity(3), np.zeros(3)

    check_non_nan(keypoints0)
    check_non_nan(keypoints1)

    F = estimate_fundamental(keypoints0, keypoints1)
    E = fundamental_to_essential(F, K)
    R1, R2, t1, t2 = extract_poses(E)

    for R_, t_ in itertools.product((R1, R2), (t1, t2)):
        X, depths_are_valid = points_from_known_poses(
            R0, R_, t0, t_, keypoints0, keypoints1, K)

        # only 1 pair (R, t) among the candidates has to be
        # the correct pair, not more nor less
        if depths_are_valid:
            return R_, t_, X

    # should not reach here
    raise ValueError("Keypoints may contain points where the depth of "
                     "the corresponding 3D point is <= 0")
