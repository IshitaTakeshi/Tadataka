import numpy as np
from tadataka.matrix import (to_homogeneous, from_homogeneous,
                             inv_motion_matrix)
from tadataka.decorator import allow_1d
from tadataka.projection import inv_pi, pi
from tadataka.pose import LocalPose, WorldPose


def warp3d(T_w0, T_w1, P0):
    Q0 = to_homogeneous(P0)
    T_1w = inv_motion_matrix(T_w1)
    T_10 = np.dot(T_1w, T_w0)
    Q1 = np.dot(T_10, Q0.T).T
    P1 = from_homogeneous(Q1)
    return P1


def warp2d(T0, T1, xs, depths):
    P = inv_pi(xs, depths)
    Q = warp3d(T0, T1, P)
    return pi(Q)


class Warp3D(object):
    def __init__(self, pose0, pose1):
        assert(isinstance(pose0, WorldPose))
        assert(isinstance(pose1, WorldPose))
        self.T0 = pose0.T
        self.T1 = pose1.T

    @allow_1d(which_argument=1)
    def __call__(self, P):
        return warp3d(self.T0, self.T1, P)


def warp_depth(warp: Warp3D, xs0, depths0):
    P0 = inv_pi(xs0, depths0)
    P1 = warp(P0)
    xs1, depths1 = pi(P1), P1[:, 2]
    return xs1, depths1


class Warp2D(object):
    """Warp coordinate between image planes"""
    def __init__(self, camera_model0, camera_model1,
                 pose0: WorldPose, pose1: WorldPose):
        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1
        self.warp3d = Warp3D(pose0, pose1)

    def __call__(self, us0, depths0):
        """
        Perform warping

        Parameters
        ----------
        us0 : (N, 2) np.ndarray
            2D coordinates in the first camera
        depths0 : (N,) np.ndarray
            Point depths in the first camera
        Returns:
        us1 : (N, 2) np.ndarray
            2D coordinates in the second camera
        depths1 : (N,) np.ndarray
            Point depths in the second camera
        """

        xs0 = self.camera_model0.normalize(us0)

        xs1, depths1 = warp_depth(self.warp3d, xs0, depths0)

        us1 = self.camera_model1.unnormalize(xs1)
        return us1, depths1


def local_warp3d_(T10, xs0, depths0):
    P0 = inv_pi(xs0, depths0)
    Q0 = to_homogeneous(P0)
    Q1 = np.dot(T10, Q0.T).T
    P1 = from_homogeneous(Q1)
    xs1, depths1 = pi(P1), P1[:, 2]
    return xs1, depths1


class LocalWarp2D(object):
    def __init__(self, camera_model0, camera_model1, pose10: LocalPose):
        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1
        self.T10 = pose10.T

    def __call__(self, us0, depths0):
        xs0 = self.camera_model0.normalize(us0)
        xs1, depths1 = local_warp3d_(self.T10, xs0, depths0)
        us1 = self.camera_model1.unnormalize(xs1)
        return us1, depths1
