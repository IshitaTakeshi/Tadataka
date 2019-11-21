import numpy as np

from tadataka.optimization.functions import Function


EPSILON = 1e-16


def pi(P):
    Z = P[:, [2]]
    XY = P[:, 0:2]
    return XY / (Z + EPSILON)


class PerspectiveProjection(object):
    def __init__(self, camera_parameters):
        self.camera_parameters = camera_parameters

    def compute(self, P):
        K = self.camera_parameters.matrix
        P = np.dot(K, P.T).T
        return pi(P)
