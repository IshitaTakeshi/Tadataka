from autograd import numpy as np

from vitamine.bundle_adjustment.mask import point_mask, keypoint_mask


def count_correspondences(points, keypoints):
    """
    Return array of shape (n_viewpoints,) which contains
    the number of correspondences between keypoints and points
    """

    masks = np.logical_and(
        point_mask(points),
        keypoint_mask(keypoints)
    )

    return np.sum(masks, axis=1)
