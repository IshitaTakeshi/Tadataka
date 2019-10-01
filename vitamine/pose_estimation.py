from autograd import numpy as np

# TODO make this independent from cv2
import cv2

from vitamine.exceptions import NotEnoughInliersException
from vitamine.so3 import rodrigues


min_correspondences = 6


def solve_pnp(points, keypoints, initial_omega=None, initial_translation=None):
    assert(points.shape[0] == keypoints.shape[0])

    if keypoints.shape[0] < min_correspondences:
        raise NotEnoughInliersException("No sufficient correspondences")

    if initial_omega is None:
        initial_omega = np.zeros(3)

    if initial_translation is None:
        initial_translation = np.zeros(3)

    assert(initial_omega.shape == (3,))
    assert(initial_translation.shape == (3,))

    retval, omega, t = cv2.solvePnP(points.astype(np.float64),
                                    keypoints.astype(np.float64),
                                    np.identity(3), np.zeros(4),
                                    initial_omega, initial_translation,
                                    useExtrinsicGuess=True,
                                    flags=cv2.SOLVEPNP_EPNP)
    return omega.flatten(), t.flatten()


def estimate_pose(points, keypoints):
    omega, t = solve_pnp(points, keypoints)
    R = rodrigues(omega.reshape(1, -1))[0]
    return R, t
