import numpy as np

import numba
from numba import njit

from tadataka.flow_estimation.regularizer import GemanMcClure
from tadataka.utils import is_in_image_range


diff_to_neighbors_ = np.array([
    [-1, -1], [0, -1], [1, -1],
    [-1, 0], [0, 0], [1, 0],
    [-1, 1], [0, 1], [1, 1]
])


@njit
def step(energy_map):
    return diff_to_neighbors_[np.argmax(energy_map)]


# regularizer term 'w'
@njit
def compute_regularizer_map(regularizer, dp):
    D = np.empty((3, 3))
    for j, y in enumerate([-1, 0, 1]):
        for i, x in enumerate([-1, 0, 1]):
            ddp = np.array([x, y])
            D[j, i] = 1 - regularizer.compute(dp + ddp)
    return D


@njit
def get_patch(curvature, p):
    px, py = p
    return curvature[py-1:py+2, px-1:px+2]


@njit
def _maximize(p0, curvature, regularizer, lambda_, max_iter):
    p = np.copy(p0)
    for i in range(max_iter):
        C = get_patch(curvature, p)
        R = compute_regularizer_map(regularizer, p - p0)
        dp = step(C + lambda_ * R)

        if dp[0] == 0 and dp[1] == 0:
            return p

        p = p + dp
    return p


@njit
def maximize(P, curvature, regularizer, lambda_, max_iter):
    for i in range(len(P)):
        P[i] = _maximize(P[i], curvature, regularizer, lambda_, max_iter)
    return P


class Maximizer(object):
    def __init__(self, curvature, regularizer, lambda_, max_iter=20):
        # fill the border with -inf
        self.curvature = np.pad(curvature, ((1, 1), (1, 1)),
                                mode="constant", constant_values=-np.inf)
        self.regularizer = regularizer
        self.lambda_ = lambda_
        self.max_iter = max_iter

    def __call__(self, P):
        offset = np.array([1, 1])
        P = P + offset
        P = maximize(P, self.curvature, self.regularizer,
                     self.lambda_, self.max_iter)
        return P - offset


# FIXME rename to 'LocalExtremaCorrection'
class ExtremaTracker(object):
    """ Optimize equation (5) """
    def __init__(self, image_curvature, lambda_,
                 regularizer=GemanMcClure(3.0)):
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
        P = self.maximizer(P)
        coordinates[mask] = P

        return coordinates + after_decimal
