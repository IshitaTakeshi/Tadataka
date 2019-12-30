import numpy as np

import numba
from numba import njit

from tadataka.flow_estimation.regularizer import get_geman_mcclure
from tadataka.utils import is_in_image_range


diff_to_neighbors_ = np.array([
    [-1, -1], [0, -1], [1, -1],
    [-1, 0], [0, 0], [1, 0],
    [-1, 1], [0, 1], [1, 1]
])


def step(energy_map):
    return diff_to_neighbors_[np.argmax(energy_map)]


# regularizer term 'w'
def compute_regularizer_map(regularizer, dp):
    xs, ys = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    DP = np.column_stack((xs.flatten(), ys.flatten()))

    D = np.array([regularizer(dp + ddp) for ddp in DP])
    D = D.reshape(3, 3)
    return 1 - D


def get_patch(curvature, p):
    px, py = p
    return curvature[py-1:py+2, px-1:px+2]


class Maximizer(object):
    def __init__(self, curvature, regularizer, lambda_, max_iter=20):
        # fill the border with -inf
        self.curvature = np.pad(curvature, ((1, 1), (1, 1)),
                                mode="constant", constant_values=-np.inf)
        self.regularizer = regularizer
        self.lambda_ = lambda_
        self.max_iter = max_iter

    def _maximize(self, p0):
        p = np.copy(p0)
        for i in range(self.max_iter):
            C = get_patch(self.curvature, p)
            R = compute_regularizer_map(self.regularizer, p - p0)
            dp = step(C + self.lambda_ * R)

            if dp[0] == 0 and dp[1] == 0:
                return p

            p = p + dp
        return p

    def __call__(self, coordinates):
        offset = np.array([1, 1])
        P = coordinates + offset
        P = self._maximize(P)
        return P - offset


# FIXME rename to 'LocalExtremaCorrection'
class ExtremaTracker(object):
    """ Optimize equation (5) """
    def __init__(self, image_curvature, lambda_,
                 regularizer=get_geman_mcclure(10.0)):
        self.maximizer = Maximizer(image_curvature, regularizer, lambda_)
        self.image_shape = image_curvature.shape

    def optimize(self, initial_coordinates):
        """
        Return corrected point coordinates
        """

        assert(np.ndim(initial_coordinates) == 2)
        assert(initial_coordinates.shape[1] == 2)

        coordinates = initial_coordinates
        coordinates = np.round(initial_coordinates)
        after_decimal = initial_coordinates - coordinates

        coordinates = coordinates.astype(np.int64)

        mask = is_in_image_range(coordinates, self.image_shape)

        P = coordinates[mask]
        for i in range(P.shape[0]):
            P[i] = self.maximizer(P[i])
        coordinates[mask] = P

        return coordinates + after_decimal
