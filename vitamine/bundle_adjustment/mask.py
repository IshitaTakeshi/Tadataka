from autograd import numpy as np


def compute_mask(array):
    assert(np.ndim(array) == 2)
    return np.all(~np.isnan(array), axis=1)


def keypoint_mask(keypoints):
    n_viewpoints, n_points = keypoints.shape[0:2]

    keypoints = keypoints.reshape(n_viewpoints * n_points, 2)
    return compute_mask(keypoints).reshape(n_viewpoints, n_points)


def pose_mask(omegas, translations):
    return np.logical_and(compute_mask(omegas), compute_mask(translations))


def point_mask(points):
    return compute_mask(points)


def correspondence_mask(keypoints1, keypoints2):
    """
    keypoints[12].shape == (n_points, 2)
    """
    return np.logical_and(
        compute_mask(keypoints1),
        compute_mask(keypoints2)
    )


def fill_masked(array, mask):
    """
    Create nan array and fill masked elements with float variables
    """
    assert(np.ndim(array) == 2)
    assert(np.ndim(mask) == 1)
    array_ = np.full((mask.shape[0], array.shape[1]), np.nan)
    array_[mask] = array
    return array_
