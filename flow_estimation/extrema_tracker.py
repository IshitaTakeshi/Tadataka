from autograd import numpy as np
from optimization.robustifiers import GemanMcClureRobustifier
from flow_estimation.image_curvature import image_curvature
from utils import is_in_image_range


class Neighbors(object):
    def __init__(self, image_shape):
        self.image_shape = image_shape[0:2]

        # 'diffs' is the coordinate differences from the center, that is
        #     [[-1, -1], [ 0, -1], [ 1, -1],
        #      [-1,  0], [ 0,  0], [ 1,  0],
        #      [-1,  1], [ 0,  1], [ 1,  1]]
        xs, ys = np.meshgrid([-1, 0, 1], [-1, 0, 1])
        self.diffs = np.vstack((xs.flatten(), ys.flatten())).T

    def get(self, p):
        """
        Return 8 neighbors of a point `p` along with `p` itself
        """
        neighbors = p + self.diffs
        mask = is_in_image_range(neighbors, self.image_shape)
        return neighbors[mask]


class Regularizer(object):
    def __init__(self, p0, robustifier=GemanMcClureRobustifier()):
        self.p0 = p0
        self.robustifier = robustifier

    def regularize(self, P):
        norms = np.linalg.norm(P - self.p0, axis=1)
        return 1 - self.robustifier.robustify(norms)


class Energy(object):
    def __init__(self, curvature, regularizer, lambda_):
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
            if (p_new == p).all():
                return p
            p = p_new
        return p


class ExtremaTracker(object):
    def __init__(self, image, keypoints, lambda_):
        self.lambda_ = lambda_
        self.curvature = image_curvature(image)
        self.keypoints = keypoints
        self.image_shape = self.curvature.shape[0:2]

    def optimize(self):
        coordinates = np.empty(self.keypoints.shape)
        for i in range(self.keypoints.shape[0]):
            p0 = self.keypoints[i]
            energy = Energy(self.curvature, Regularizer(p0), self.lambda_)
            maximizer = Maximizer(energy, self.image_shape)
            coordinates[i] = maximizer.search(p0)
        return coordinates
