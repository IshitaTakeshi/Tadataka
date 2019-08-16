from vitamine.bundle_adjustment.mask import keypoint_mask, point_mask, compute_mask

from autograd import numpy as np


def solve_pnp(points, keypoints, K):
    # TODO make independent from cv2
    import cv2

    retval, rvec, tvec = cv2.solvePnP(points.astype(np.float64),
                                      keypoints.astype(np.float64),
                                      K, np.zeros(4))
    rvec = rvec.flatten()
    tvec = tvec.flatten()
    return rvec, tvec


def estimate_pose(points, keypoints, K):
    # keypoints.shape == (n_points, 2)
    assert(keypoints.shape[0] == points.shape[0])
    assert(keypoints.shape[1] == 2)
    assert(points.shape[1] == 3)

    # at least 'min_correspondences' corresponding points
    # have to be found between keypoitns and 3D poitns
    # to perform PnP

    min_correspondences = 4

    # mask indicates which 3D point to ignore / use from each viewpoint
    mask = np.logical_and(
        point_mask(points),
        compute_mask(keypoints)
    )

    if np.sum(mask) < min_correspondences:
        return np.full(3, np.nan), np.full(3, np.nan)

    # use only non nan elements to perform PnP
    X, P = points[mask], keypoints[mask]
    return solve_pnp(X, P, K)


def estimate_poses(points, keypoints, K):
    n_viewpoints = keypoints.shape[0]

    omegas = np.empty((n_viewpoints, 3))
    translations = np.empty((n_viewpoints, 3))

    for i, keypoints_ in enumerate(keypoints):
        omegas[i], translations[i] = estimate_pose(points, keypoints_, K)
    return omegas, translations
