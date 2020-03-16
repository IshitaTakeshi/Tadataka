from numba import njit
import numpy as np

from tadataka.matrix import to_homogeneous


EPSILON = 1e-16


@njit
def pi(P):
    if P.ndim == 1:
        return P[0:2] / (P[2] + EPSILON)

    # copy since numba doesn't support reshape of non-contiguous array
    z = np.copy(P[:, 2])
    Z = z.reshape(z.shape[0], 1)

    XY = P[:, 0:2]

    return XY / (Z + EPSILON)


@njit
def inv_pi(xs, depths):
    if xs.ndim == 1:
        return depths * to_homogeneous(xs)
    return depths.reshape(-1, 1) * to_homogeneous(xs)


class PerspectiveProjection(object):
    def __init__(self, camera_parameters):
        self.camera_parameters = camera_parameters

    def compute(self, P):
        K = self.camera_parameters.matrix
        P = np.dot(K, P.T).T
        return pi(P)
