from autograd import numpy as np

# TODO make this independent from cv2
import cv2

from vitamine.exceptions import NotEnoughInliersException
from vitamine.so3 import exp_so3, log_so3


class Pose(object):
    def __init__(self, R_or_omega, t):
        if np.ndim(R_or_omega) == 1:
            self.omega = R_or_omega
        elif np.ndim(R_or_omega) == 2:
            self.omega = log_so3(R_or_omega)

        self.t = t

    @property
    def R(self):
        return exp_so3(self.omega)

    def __str__(self):
        with np.printoptions(precision=3, suppress=True):
            return "omega = " + str(self.omega)  + "   t = " + str(self.t)

    @staticmethod
    def identity():
        return Pose(np.zeros(3), np.zeros(3))

    def __eq__(self, other):
        return (np.isclose(self.omega, other.omega).all() and
                np.isclose(self.t, other.t).all())


min_correspondences = 6


def calc_reprojection_threshold(keypoints, k=2.0):
    center = np.mean(keypoints, axis=0, keepdims=True)
    squared_distances = np.sum(np.power(keypoints - center, 2), axis=1)
    # rms of distances from center to keypoints
    rms = np.sqrt(np.mean(squared_distances))
    return k * rms / keypoints.shape[0]


def solve_pnp(points, keypoints):
    assert(points.shape[0] == keypoints.shape[0])

    if keypoints.shape[0] < min_correspondences:
        raise NotEnoughInliersException("No sufficient correspondences")

    print("keypoints.shape", keypoints.shape)
    t = calc_reprojection_threshold(keypoints, k=3.0)
    print("reprojectionError", t)
    retval, omega, t, inliers = cv2.solvePnPRansac(
        points.astype(np.float64),
        keypoints.astype(np.float64),
        np.identity(3), np.zeros(4),
        reprojectionError=t,
        flags=cv2.SOLVEPNP_EPNP
    )
    print("retval: {}".format(retval))
    print("inlier ratio")
    print(len(inliers.flatten()) / points.shape[0])

    if len(inliers.flatten()) == 0:
        raise NotEnoughInliersException("No inliers found")

    return Pose(omega.flatten(), t.flatten())
