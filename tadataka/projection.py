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
    def __init__(self, camera_model0, camera_model1, local_pose01):
        if not isinstance(local_pose01, LocalPose):
            raise ValueError("Pose must be an instance of LocalPose")
        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1
        self.pose01 = local_pose01

    def __call__(self, us0, depths0):
        xs0 = self.camera_model0.normalize(us0)
        xs1 = warp(xs0, depths0, self.pose01.R, self.pose01.t)
        us1 = self.camera_model1.unnormalize(xs1)
        us1 = np.round(us1).astype(np.int64)
        return us1
