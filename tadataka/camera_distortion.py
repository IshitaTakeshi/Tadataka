# reference
# https://github.com/colmap/colmap/blob/master/src/base/camera_models.h

from autograd import numpy as np


EPSILON = 1e-4


class Normalizer(object):
    def __init__(self, camera_parameters):
        self.focal_length = camera_parameters.focal_length
        self.offset = camera_parameters.offset

    def normalize(self, keypoints):
        """
        Transform keypoints to the normalized plane
        (x - cx) / fx = X / Z
        (y - cy) / fy = Y / Z
        """
        return (keypoints - self.offset) / self.focal_length

    def inverse(self, normalized_keypoints):
        """
        Inverse transformation from the normalized plane
        x = fx * X / Z + cx
        y = fy * Y / Z + cy
        """
        return normalized_keypoints * self.focal_length + self.offset


def distort_factors(X, omega):
    def f(r):
        return np.arctan(2 * r * np.tan(omega / 2)) / omega

    r = np.linalg.norm(X, axis=1)

    iszero = np.isclose(r, 0)

    factors = np.empty(r.shape)
    factors[iszero] = 2 * np.tan(omega / 2) / omega  # linear approx of f
    factors[~iszero] = f(r[~iszero])
    return factors


def undistort_factors(X, omega):
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

        factors = distort_factors(undistorted_keypoints, self.omega)
        return factors.reshape(-1, 1) * undistorted_keypoints

    def undistort(self, distorted_keypoints):
        if np.isclose(self.omega, 0):
            return distorted_keypoints  # all factors = 1

        factors = undistort_factors(distorted_keypoints, self.omega)
        return factors.reshape(-1, 1) * distorted_keypoints


class CameraModel(object):
    def __init__(self, camera_parameters, distortion_model):
        self.normalizer = Normalizer(camera_parameters)
        self.distortion_model = distortion_model

    def undistort(self, keypoints):
        return self.distortion_model.undistort(
            self.normalizer.normalize(keypoints)
        )

    def distort(self, normalized_keypoints):
        return self.normalizer.inverse(
            self.distortion_model.undistort(normalized_keypoints)
        )
