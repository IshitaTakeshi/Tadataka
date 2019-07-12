from autograd import numpy as np


class ParameterConverter(object):
    def __init__(self, n_viewpoints, n_points):
        self.n_viewpoints = n_viewpoints
        self.n_points = n_points

    def from_params(self, params):
        assert(params.shape[0] == self.n_viewpoints * 6 + self.n_points * 3)

        N = 3 * self.n_viewpoints
        omegas = params[0:N].reshape(self.n_viewpoints, 3)
        translations = params[N:2*N].reshape(self.n_viewpoints, 3)
        points = params[2*N:].reshape(self.n_points, 3)

        return omegas, translations, points

    def to_params(self, omegas, translations, points):
        return np.concatenate((
            omegas.flatten(),
            translations.flatten(),
            points.flatten()
        ))

    @property
    def n_dims(self):
        return 6 * self.n_viewpoints + 3 * self.n_points
