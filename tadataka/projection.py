import numpy as np

from tadataka.optimization.functions import Function
from tadataka.matrix import to_homogeneous
from tadataka.rigid_transform import transform


EPSILON = 1e-16


def pi(P):
    if np.ndim(P) == 1:
        return P[0:2] / (P[2] + EPSILON)

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


def warp(coordinates, depths, R, t):
    P = to_homogeneous(coordinates) * depths.reshape(-1, 1)
    return pi(transform(R, t, P))
