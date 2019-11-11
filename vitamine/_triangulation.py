import itertools

from autograd import numpy as np
from skimage.transform import ProjectiveTransform, FundamentalMatrixTransform

from vitamine.matrix import solve_linear, motion_matrix
from vitamine.so3 import rodrigues


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


def fundamental_to_essential(F, K0, K1=None):
    if K1 is None:
        K1 = K0
    return K1.T.dot(F).dot(K0)


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


def depths_are_valid(depth0, depth1, min_depth):
    return depth0 > min_depth and depth1 > min_depth


def points_from_known_poses(R0, R1, t0, t1, keypoints0, keypoints1,
                            min_depth=0.0):
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

    depth_mask = np.zeros(n_points, dtype=np.bool)

    for i in range(n_points):
        points[i], depth0, depth1 = linear_triangulation(
            R0, R1, t0, t1, keypoints0[i], keypoints1[i])
        depth_mask[i] = depths_are_valid(depth0, depth1, min_depth)
    return points, depth_mask


def estimate_homography(keypoints1, keypoints2):
    tform = ProjectiveTransform()
    tform.estimate(keypoints1, keypoints2)
    return tform.params


def estimate_fundamental(keypoints1, keypoints2):
    tform = FundamentalMatrixTransform()
    tform.estimate(keypoints1, keypoints2)
    return tform.params


def n_triangulated(n_keypoints, triangulation_ratio=0.2, n_min_triangulation=40):
    n = int(n_keypoints * triangulation_ratio)
    # at least use 'n_min_triangulation' points
    k = max(n, n_min_triangulation)
    # make the return value not exceed the number of keypoints
    return min(n_keypoints, k)


def triangulation_indices(n_keypoints):
    N = n_triangulated(n_keypoints)
    indices = np.arange(0, n_keypoints)
    np.random.shuffle(indices)
    return indices[:N]


def estimate_camera_pose_change(keypoints0, keypoints1):
    """
    Reconstruct 3D points from non nan keypoints obtained from 2 views
    keypoints[01].shape == (n_points, 2)
    keypoints0 and keypoints1 are undistorted and normalized keypoints
    """

    assert(keypoints0.shape == keypoints1.shape)

    R0, t0 = np.identity(3), np.zeros(3)

    # we assume that keypoints are normalized
    E = estimate_fundamental(keypoints0, keypoints1)

    # R <- {R1, R2}, t <- {t1, t2} satisfy
    # K * [R | t] * homegeneous(points) = homogeneous(keypoint)
    R1, R2, t1, t2 = extract_poses(E)
    n_max_valid_depth = -1
    argmax_R, argmax_t = None, None

    # not necessary to triangulate all points to validate depths
    indices = triangulation_indices(len(keypoints0))
    for i, (R_, t_) in enumerate(itertools.product((R1, R2), (t1, t2))):
        _, depth_mask = points_from_known_poses(
            R0, R_, t0, t_, keypoints0[indices], keypoints1[indices])
        n_valid_depth = np.sum(depth_mask)

        # only 1 pair (R, t) among the candidates has to be
        # the correct pair
        if n_valid_depth > n_max_valid_depth:
            n_max_valid_depth = n_valid_depth
            argmax_R, argmax_t = R_, t_
    return argmax_R, argmax_t
