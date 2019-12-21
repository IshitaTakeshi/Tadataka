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


# regularizer term 'w'
def compute_regularizer_map(regularizer):
    D = np.empty((3, 3))
    for y in range(3):
        for x in range(3):
            D[y, x] = regularizer(diff_to_neighbors_[y * 3 + x])
    return 1 - D


@njit
def step(energy_map):
    return diff_to_neighbors_[np.argmax(energy_map)]


def maximize_one(curvature, regularizer_map, p0, max_iter=20):
    px, py = p0
    for i in range(max_iter):
        C = curvature[py-1:py+2, px-1:px+2]

        dpx, dpy = step(C + regularizer_map)

        if dpx == 0 and dpy == 0:
            return [px, py]

        px, py = px + dpx, py + dpy
    return [px, py]


def maximize_(curvature, regularizer_map, coordinates):
    for i in range(coordinates.shape[0]):
        coordinates[i] = maximize_one(curvature, regularizer_map,
                                      coordinates[i])
    return coordinates


def maximize(curvature, regularizer_map, coordinates):
    C = np.pad(curvature, ((1, 1), (1, 1)),
               mode="constant", constant_values=-np.inf)
    offset = np.array([1, 1])
    P = coordinates + offset

    P = maximize_(C, regularizer_map, P)

    return P - offset


# FIXME rename to 'LocalExtremaCorrection'
class ExtremaTracker(object):
    """ Optimize equation (5) """
    def __init__(self, image_curvature, lambda_,
                 regularizer=get_geman_mcclure(1.0)):
        self.curvature = image_curvature

        R = compute_regularizer_map(regularizer)
        self.regularizer_map = lambda_ * R

    def optimize(self, initial_coordinates):
        """
        Return corrected point coordinates
        """

        assert(np.ndim(initial_coordinates) == 2)
        assert(initial_coordinates.shape[1] == 2)

        coordinates = np.round(initial_coordinates)
        after_decimal = initial_coordinates - coordinates

        coordinates = coordinates.astype(np.int64)

        mask = is_in_image_range(coordinates, self.curvature.shape)

        coordinates[mask] = maximize(self.curvature, self.regularizer_map,
                                     coordinates[mask])

        return coordinates.astype(np.float64) + after_decimal
