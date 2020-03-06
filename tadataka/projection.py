import numpy as np

from tadataka.optimization.functions import Function
from tadataka.matrix import to_homogeneous
from tadataka.rigid_transform import transform
from tadataka.pose import LocalPose

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
    """
    Warp from a normalized image plane to the other normalized image plane
    """
    P = to_homogeneous(coordinates) * depths.reshape(-1, 1)
    return pi(transform(R, t, P))


class Warp(object):
    def __init__(self, camera_model0, camera_model1, pose0, pose1):
        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1
        self.pose0 = pose0
        self.pose1 = pose1

    def __call__(self, u0, depth0):
        x0 = self.camera_model0.normalize(u0)
        p = depth0 * np.append(x0, 1)
        p = np.dot(self.pose0.R, p) + self.pose0.t
        p = np.dot(self.pose1.R.T, p - self.pose1.t)
        x1 = pi(p)
        u1 = self.camera_model1.unnormalize(x1)
        return u1
