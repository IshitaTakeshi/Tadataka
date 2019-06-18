from autograd import numpy as np
from optimization.robustifiers import GemanMcClureRobustifier
from flow_estimation.image_curvature import curvature
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
        self.robustifier = robustifier
        self.p0 = p0

    def regularize(self, P):
        norms = np.linalg.norm(P - self.p0)
        return 1 - self.robustifier.robustify(norms)


class Energy(object):
    def __init__(self, K, regularizer, lambda_=0.3):
        self.K = K
        self.regularizer = regularizer
        self.lambda_ = lambda_

    def compute(self, coordinates):
        R = self.regularizer.regularize(coordinates)
        xs, ys = coordinates[:, 0], coordinates[:, 1]
        return self.K[ys, xs] + self.lambda_ * R


class Maximizer(object):
    def __init__(self, image, p0, n_max_iter=20):
        self.p0 = p0
        self.energy = Energy(curvature(image), Regularizer(p0))
        self.neighbors = Neighbors(image.shape)
        self.n_max_iter = n_max_iter

    def search_neighbors(self, p):
        neighbors = self.neighbors.get(p)
        argmax = np.argmax(self.energy.compute(neighbors))
        return neighbors[argmax]

    def search(self):
        p = np.copy(self.p0)
        for i in range(self.n_max_iter):
            p = self.search_neighbors(p)
        return p


class ExtremaTracker(object):
    def __init__(self, image, keypoints):
        self.image = image
        self.keypoints = keypoints

    def optimize(self):
        coordinates = np.empty(self.keypoints.shape)
        for i in range(self.keypoints.shape[0]):
            maximizer = Maximizer(self.image, self.keypoints[i])
            coordinates[i] = maximizer.search()
        return coordinates
