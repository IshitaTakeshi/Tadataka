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
