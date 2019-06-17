from autograd import numpy as np

from flow_estimation.image_curvature import curvature


class ExtremaTracker(object):
    def __init__(self, image, keypoints, regularizer,
                 n_max_iter=20, lambda_=0.3):
        self.K = curvature(image)
        self.keypoints = keypoints
        self.regularizer = regularizer
        self.n_max_iter = n_max_iter
        self.lambda_ = lambda_

        # coordinate differences from the center
        # [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], ..., [1, 1]]
        xs, ys = np.meshgrid([-1, 0, 1], [-1, 0, 1])
        self.diff_to_neighbors = np.vstack((xs.flatten(), ys.flatten())).T

    def energy(self, P, p0):
        R = self.regularizer(np.linalg.norm(P - p0, 1))
        xs, ys = P[:, 0], P[:, 1]
        return self.K[ys, xs] + self.lambda_ * R

    def search_neighbors(self, p, p0):
        neighbors = p + self.diff_to_neighbors
        argmax = np.argmax(self.energy(neighbors, p0))
        return neighbors[argmax]

    def search(self, p0):
        p = np.copy(p0)
        for i in range(self.n_max_iter):
            p = self.search_neighbors(p, p0)
        return p

    def optimize(self):
        coordinates = np.empty(self.keypoints.shape)
        for i in range(self.keypoints.shape[0]):
            coordinates[i] = self.search(self.keypoints[i])
        return coordinates
