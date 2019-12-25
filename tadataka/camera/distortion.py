# reference
# https://github.com/colmap/colmap/blob/master/src/base/camera_models.h

import numpy as np


EPSILON = 1e-4


def fov_distort_factors(X, omega):
    def f(r):
        return np.arctan(2 * r * np.tan(omega / 2)) / omega

    r = np.linalg.norm(X, axis=1)

    iszero = np.isclose(r, 0)

    factors = np.empty(r.shape)
    factors[iszero] = 2 * np.tan(omega / 2) / omega  # linear approx of f
    factors[~iszero] = f(r[~iszero])
    return factors


def fov_undistort_factors(X, omega):
    def f(r):
        return np.tan(r * omega) / (2 * r * np.tan(omega / 2))

    r = np.linalg.norm(X, axis=1)

    iszero = np.isclose(r, 0)

    factors = np.empty(r.shape)
    factors[iszero] = omega / (2 * np.tan(omega / 2))  # linear approx of f
    factors[~iszero] = f(r[~iszero])
    return factors


class FOV(object):
    """
    Devernay, Frederic, and Olivier D. Faugeras.
    "Automatic calibration and removal of distortion from
     scenes of structured environments."
    Investigative and Trial Image Processing. Vol. 2567.
    International Society for Optics and Photonics, 1995.
    """
    def __init__(self, omega):
        self.omega = omega

    def distort(self, undistorted_keypoints):
        if np.isclose(self.omega, 0):
            return undistorted_keypoints

        factors = fov_distort_factors(undistorted_keypoints, self.omega)
        return factors.reshape(-1, 1) * undistorted_keypoints

    def undistort(self, distorted_keypoints):
        if np.isclose(self.omega, 0):
            return distorted_keypoints  # all factors = 1

        factors = fov_undistort_factors(distorted_keypoints, self.omega)
        return factors.reshape(-1, 1) * distorted_keypoints

    @staticmethod
    def from_params(params):
        assert(len(params) == 1)
        return FOV(omega=params[0])

    @property
    def params(self):
        return [self.omega]


class RadTan(object):
    def __init__(self, dist_coeffs):
        self.dist_coeffs = dist_coeffs

    def dostort(self):
        raise NotImplementedError()

    def undistort(self, distorted_keypoints):
        import cv2
        K = np.identity(3)
        keypoints = cv2.undistortPoints(distorted_keypoints, K, np.zeros(4))
        return keypoints.reshape(distorted_keypoints.shape)

    @staticmethod
    def from_params(params):
        return RadTan(params)

    @property
    def params(self):
        return self.dist_coeffs
