import numpy as np

from tadataka.utils import is_in_image_range


def diff_to_neighbors():
    # return the coordinate differences from the center, that is
    #     [[-1, -1], [ 0, -1], [ 1, -1],
    #      [-1,  0], [ 0,  0], [ 1,  0],
    #      [-1,  1], [ 0,  1], [ 1,  1]]
    xs, ys = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    return np.vstack((xs.flatten(), ys.flatten())).T


diff_to_neighbors_ = diff_to_neighbors()


class Neighbors(object):
    def __init__(self, image_shape):
        self.image_shape = image_shape[0:2]

    def get(self, p):
        """
        Return 8 neighbors of a point `p` along with `p` itself
        """
        neighbors = p + diff_to_neighbors_
        mask = is_in_image_range(neighbors, self.image_shape)
        return neighbors[mask]


class Regularizer(object):
    def __init__(self, p0):
        self.p0 = p0

    def regularize(self, P):
        return 1 - np.sum(np.power(P - self.p0, 2), axis=1)
        # norms = np.linalg.norm(P - self.p0, axis=1)
        # return 1. - np.power(norms, 2)


class Energy(object):
    def __init__(self, curvature, regularizer, lambda_):
        assert(np.ndim(curvature) == 2)
        self.curvature = curvature
        self.regularizer = regularizer
        self.lambda_ = lambda_

    def compute(self, coordinates):
        R = self.regularizer.regularize(coordinates)
        xs, ys = coordinates[:, 0], coordinates[:, 1]
        return self.curvature[ys, xs] + self.lambda_ * R


class Maximizer(object):
    def __init__(self, energy, image_shape, max_iter=20):
        self.energy = energy
        self.neighbors = Neighbors(image_shape)
        self.max_iter = max_iter

    def search_neighbors(self, p):
        neighbors = self.neighbors.get(p)
        argmax = np.argmax(self.energy.compute(neighbors))
        return neighbors[argmax]

    def search(self, p0):
        p = np.copy(p0)
        for i in range(self.max_iter):
            p_new = self.search_neighbors(p)
            if (p_new == p).all():  # converged
                return p
            p = p_new
        return p


def isint(array):
    return array.dtype.kind == 'i'


# FIXME rename to 'LocalExtremaCorrection'
class ExtremaTracker(object):
    """ Optimize equation (5) """
    def __init__(self, image_curvature, lambda_):
        self.curvature = image_curvature
        self.image_shape = self.curvature.shape[0:2]

        self.lambda_ = lambda_

    def optimize(self, initial_coordinates):
        """
        Return corrected point coordinates
        """

        print("initial_coordinates.shape", initial_coordinates.shape)

        initial_coordinates = initial_coordinates.astype(np.int64)
        assert(is_in_image_range(initial_coordinates, self.image_shape).all())

        coordinates = np.empty(initial_coordinates.shape, dtype=np.int64)
        for i in range(initial_coordinates.shape[0]):
            p0 = initial_coordinates[i]
            reguralizer = Regularizer(p0)
            energy = Energy(self.curvature, reguralizer, self.lambda_)
            maximizer = Maximizer(energy, self.image_shape)
            coordinates[i] = maximizer.search(p0)
        return coordinates
