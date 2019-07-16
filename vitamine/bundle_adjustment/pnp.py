from vitamine.bundle_adjustment.mask import keypoint_mask, point_mask

from autograd import numpy as np


def solve_pnp(points, keypoints, K):
    # TODO make independent from cv2
    import cv2

    retval, rvec, tvec = cv2.solvePnP(points, keypoints, K, np.zeros(4))
    rvec = rvec.flatten()
    tvec = tvec.flatten()
    return rvec, tvec


def estimate_poses(points, keypoints, K):
    # at least 'min_correspondences' corresponding points
    # have to be found between keypoitns and 3D poitns
    # to perform PnP
    min_correspondences = 4

    # masks.shape == (n_viewpoints, points)
    # indicates which 3D point to ignore / use from each viewpoint
    masks = np.logical_and(
        point_mask(points),
        keypoint_mask(keypoints)
    )

    n_viewpoints = keypoints.shape[0]

    omegas = np.empty((n_viewpoints, 3))
    translations = np.empty((n_viewpoints, 3))
    for i in range(n_viewpoints):
        mask = masks[i]

        if np.sum(mask) < min_correspondences:
            omegas[i] = np.nan
            translations[i] = np.nan
            continue

        omegas[i], translations[i] = solve_pnp(
            points[mask], keypoints[i, mask], K)
    return omegas, translations
