import numpy as np

from tadataka.matrix import to_homogeneous
from rust_bindings import projection


EPSILON = 1e-16


def pi(P):
    """
    Project 3D points onto normalized image plane
    """
    if P.ndim == 1:
        return projection.project_vec(P)
    return projection.project_vecs(P)


def inv_pi(xs, depths):
    """
    Inverse projection from normalized image plane to 3D
    """
    if xs.ndim == 1:
        return projection.inv_project_vec(xs, depths)
    return projection.inv_project_vecs(xs, depths)


class PerspectiveProjection(object):
    def __init__(self, camera_parameters):
        self.camera_parameters = camera_parameters

    def compute(self, P):
        K = self.camera_parameters.matrix
        P = np.dot(K, P.T).T
        return pi(P)
