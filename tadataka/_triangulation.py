import numpy as np

from tadataka.matrix import solve_linear


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
        keypoints : np.ndarray (n_keypoints, n_poses, 2)
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
