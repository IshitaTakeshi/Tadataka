# reference
# https://github.com/colmap/colmap/blob/master/src/base/camera_models.h

from autograd import numpy as np


EPSILON = 1e-4


class Normalizer(object):
    def __init__(self, camera_parameters):
        self.focal_length = camera_parameters.focal_length
        self.offset = camera_parameters.offset

    def normalize(self, keypoints):
        return (keypoints - self.offset) / self.focal_length


def calc_factors(X, omega):
    f = lambda r: np.tan(r * omega) / (2 * r * np.tan(omega / 2))
    r = np.linalg.norm(X, axis=1)
    mask = np.isclose(r, 0)
    factors = np.empty(r.shape)
    factors[mask] = omega / (2 * np.tan(omega / 2))
    factors[~mask] = f(r[~mask])
    return factors


class FOV(object):
    def __init__(self, omega):
        self.omega = omega

    def distort(self):
        # TODO
        pass

    def undistort(self, normalized_keypoints):
        if np.isclose(self.omega, 0):
            return normalized_keypoints  # all factors = 1

        factors = calc_factors(normalized_keypoints, self.omega)
        return factors.reshape(-1, 1) * normalized_keypoints


class CameraModel(object):
    def __init__(self, camera_parameters, distortion_model):
        self.normalizer = Normalizer(camera_parameters)
        self.distortion_model = distortion_model

    def undistort(self, keypoints):
        normalized = self.normalizer.normalize(keypoints)
        return self.distortion_model.undistort(normalized)
