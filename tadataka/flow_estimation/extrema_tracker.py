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


@njit
def step(energy_map):
    return diff_to_neighbors_[np.argmax(energy_map)]


def compute_image_coordinates(image_shape):
    height, width = image_shape
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    return np.column_stack((xs.flatten(), ys.flatten()))


# regularizer term 'w'
def compute_regularizer_map(regularizer, image_shape, p0):
    C = compute_image_coordinates(image_shape)
    D = regularizer(C - p0).reshape(image_shape)
    return 1 - D

@njit
def get_patch(curvature, p):
    px, py = p
    return curvature[py-1:py+2, px-1:px+2]


@njit
def search_(C, p, max_iter):
    offset = np.array([1, 1])
    p = p + offset
    for i in range(max_iter):
        dp = step(get_patch(C, p))
        p = p + dp
    return p - offset


def search(C, p, max_iter):
    C = np.pad(C, ((1, 1), (1, 1)),
               mode="constant", constant_values=-np.inf)
    return search_(C, p, max_iter)


class Maximizer(object):
    def __init__(self, curvature, regularizer, lambda_, max_iter=20):
        # fill the border with -inf
        self.curvature = curvature
        self.regularizer = regularizer
        self.lambda_ = lambda_
        self.max_iter = max_iter

    def __call__(self, p0):
        image_shape = self.curvature.shape
        R = compute_regularizer_map(self.regularizer,
                                    image_shape, p0)
        C = self.curvature + self.lambda_ * R
        p = search(C, p0, self.max_iter)
        # y, x = np.unravel_index(np.argmax(C), image_shape)
        # p = [x, y]
        return p

        from matplotlib import pyplot as plt

        plt.subplot(221)
        plt.imshow(self.curvature, cmap="gray")
        plt.xlim([0, image_shape[1]])
        plt.ylim([image_shape[0], 0])
        plt.scatter(p0[0], p0[1])

        plt.subplot(222)
        plt.imshow(R, cmap="gray")
        plt.xlim([0, image_shape[1]])
        plt.ylim([image_shape[0], 0])
        plt.scatter(p0[0], p0[1])

        plt.subplot(223)
        plt.imshow(C, cmap="gray")
        plt.xlim([0, image_shape[1]])
        plt.ylim([image_shape[0], 0])
        plt.scatter(p[0], p[1])

        plt.show()
        return p


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
