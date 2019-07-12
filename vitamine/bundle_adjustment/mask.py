from autograd import numpy as np


def mask(array):
    assert(np.ndim(array) == 2)
    return np.all(~np.isnan(array), axis=1)


def keypoint_mask(keypoints):
    n_viewpoints, n_points = keypoints.shape[0:2]

    keypoints = keypoints.reshape(n_viewpoints * n_points, 2)
    return mask(keypoints).reshape(n_viewpoints, n_points)


def pose_mask(omegas, translations):
    return np.logical_and(mask(omegas), mask(translations))


def point_mask(points):
    return mask(points)


def fill_masked(array, mask):
    """
    Create nan array and fill masked elements with float variables
    """
    assert(np.ndim(array) == 2)
    assert(np.ndim(mask) == 1)
    array_ = np.full((mask.shape[0], array.shape[1]), np.nan)
    array_[mask] = array
    return array_
