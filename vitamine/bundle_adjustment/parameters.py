from autograd import numpy as np

from vitamine.bundle_adjustment.mask import pose_mask, point_mask, fill_masked


class ParameterConverter(object):
    def to_params(self, omegas, translations, points):
        self.pose_mask = pose_mask(omegas, translations)
        self.point_mask = point_mask(points)

        return np.concatenate((
            omegas[self.pose_mask].flatten(),
            translations[self.pose_mask].flatten(),
            points[self.point_mask].flatten()
        ))

    def mask_keypoints(self, keypoints):
        """
        Remove keypoint elements where the estimated projection becomes nan
        """
        # FIXME just not clever
        keypoints = keypoints[self.pose_mask]
        keypoints = keypoints[:, self.point_mask]
        return keypoints

    def from_params(self, params):
        assert(params.shape[0] == self.ndim)

        N = self.n_valid_viewpoints
        M = self.n_valid_points

        omegas = params[0:3*N].reshape((N, 3))
        translations = params[3*N:6*N].reshape((N, 3))
        points = params[6*N:6*N+3*M].reshape((M, 3))

        return omegas, translations, points

    @property
    def n_valid_viewpoints(self):
        return np.sum(self.pose_mask)

    @property
    def n_valid_points(self):
        return np.sum(self.point_mask)

    @property
    def ndim(self):
        return self.n_valid_viewpoints * 6 + self.n_valid_points * 3
