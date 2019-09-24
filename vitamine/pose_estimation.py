from autograd import numpy as np

from vitamine.exceptions import NotEnoughInliersException

# TODO make this independent from cv2
import cv2


min_correspondences = 4


def solve_pnp(points, keypoints):
    assert(points.shape[0] == keypoints.shape[0])

    if keypoints.shape[0] < min_correspondences:
        raise NotEnoughInliersException("No sufficient correspondences")

    retval, omega, t = cv2.solvePnP(points.astype(np.float64),
                                    keypoints.astype(np.float64),
                                    np.identity(3), np.zeros(4))
    return omega.flatten(), t.flatten()
