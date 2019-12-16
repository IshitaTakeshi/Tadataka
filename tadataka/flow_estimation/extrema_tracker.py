import numpy as np

import numba
from numba import jitclass, njit


def diff_to_neighbors():
    # return the coordinate differences from the center, that is
    #     [[-1, -1], [ 0, -1], [ 1, -1],
    #      [-1,  0], [ 0,  0], [ 1,  0],
    #      [-1,  1], [ 0,  1], [ 1,  1]]
    xs, ys = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    return np.vstack((xs.flatten(), ys.flatten())).T


diff_to_neighbors_ = diff_to_neighbors()


@njit
def get_neighbors(p, image_shape):
    """
    Return 8 neighbors of a point `p` along with `p` itself
    """
    neighbors = p + diff_to_neighbors_
    mask = is_in_image_range(neighbors, image_shape)
    return neighbors[mask]


@njit
def is_in_image_range(P, image_shape):
    height, width = image_shape[0:2]
    xs, ys = P[:, 0], P[:, 1]
    mask_x = np.logical_and(0 <= xs, xs < width)
    mask_y = np.logical_and(0 <= ys, ys < height)
    return np.logical_and(mask_x, mask_y)


@njit
def regularize(p, p0):
    return 1. - np.sum(np.power(p - p0, 2))


@njit
def energy(curvature, p0, P, lambda_):
    N = len(P)
    E = np.empty(N, dtype=np.float64)
    for i in range(N):
        p = P[i]
        x, y = p
        E[i] = curvature[y, x] + lambda_ * regularize(p, p0)
    return E


@njit
def search_neighbors(curvature, p0, p, lambda_):
    neighbors = get_neighbors(p, curvature.shape)
    E = energy(curvature, p0, neighbors, lambda_)
    return neighbors[np.argmax(E)]


@njit
def maximize_(curvature, p0, lambda_, max_iter=20):
    p = np.copy(p0)
    for i in range(max_iter):
        p_new = search_neighbors(curvature, p0, p, lambda_)
        if (p_new == p).all():
            # converged
            return p
        p = p_new
    return p


@njit
def maximize(curvature, coordinates, lambda_):
    for i in range(coordinates.shape[0]):
        coordinates[i] = maximize_(curvature, coordinates[i], lambda_)
    return coordinates


# FIXME rename to 'LocalExtremaCorrection'
class ExtremaTracker(object):
    """ Optimize equation (5) """
    def __init__(self, image_curvature, lambda_):
        self.curvature = image_curvature

        self.lambda_ = lambda_

    def optimize(self, initial_coordinates):
        """
        Return corrected point coordinates
        """

        assert(np.ndim(initial_coordinates) == 2)
        assert(initial_coordinates.shape[1] == 2)

        coordinates = initial_coordinates.astype(np.int64)
        image_shape = self.curvature.shape
        assert(is_in_image_range(coordinates, image_shape).all())

        return maximize(self.curvature, coordinates, self.lambda_)
