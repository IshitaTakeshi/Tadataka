from autograd import numpy as np

from vitamine.bundle_adjustment.mask import pose_mask, point_mask, fill_masked


def to_params(omegas, translations, points):
    return np.concatenate((
        omegas.flatten(),
        translations.flatten(),
        points.flatten()
    ))


def from_params(params, n_valid_viewpoints, n_valid_points):
    N = n_valid_viewpoints
    M = n_valid_points

    assert(params.shape[0] == N * 6 + M * 3)

    omegas = params[0:3*N].reshape((N, 3))
    translations = params[3*N:6*N].reshape((N, 3))
    points = params[6*N:6*N+3*M].reshape((M, 3))

    return omegas, translations, points


class ParameterMask(object):
    def __init__(self, omegas, translations, points):
        self.omegas = omegas
        self.translations = translations
        self.points = points

        self.pose_mask = pose_mask(omegas, translations)
        self.point_mask = point_mask(points)

    def get_masked(self):
        omegas, translations = self.mask_poses(self.omegas, self.translations)
        points = self.mask_points(self.points)
        return omegas, translations, points

    def mask_poses(self, omegas, translations):
        return omegas[self.pose_mask], translations[self.pose_mask]

    def mask_points(self, points):
        return points[self.point_mask]

    def mask_keypoints(self, keypoints):
        keypoints = keypoints[self.pose_mask]
        keypoints = keypoints[:, self.point_mask]
        return keypoints

    def fill(self, omegas, translations, points):
        omegas = fill_masked(omegas, self.pose_mask)
        translations = fill_masked(translations, self.pose_mask)
        points = fill_masked(points, self.point_mask)
        return omegas, translations, points

    @property
    def n_valid_viewpoints(self):
        return np.sum(self.pose_mask)

    @property
    def n_valid_points(self):
        return np.sum(self.point_mask)
