import numpy as np

from tadataka.matrix import to_homogeneous
from tadataka import _projection


EPSILON = 1e-16


def pi(P):
    """
    Project 3D points onto normalized image plane
    """
    if P.ndim == 1:
        u = np.empty(2)
        _projection.project_vector(P, u)
        return u

    U = np.empty((P.shape[0], 2))
    _projection.project_vectors(P, U)
    return U



def inv_pi(xs, depths):
    """
    Inverse projection from normalized image plane to 3D
    """
    return _projection.inv_pi(xs, depths)


class PerspectiveProjection(object):
    def __init__(self, camera_parameters):
        self.camera_parameters = camera_parameters

    def compute(self, P):
        K = self.camera_parameters.matrix
        P = np.dot(K, P.T).T
        return pi(P)
