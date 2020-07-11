import numpy as np
from tadataka.matrix import calc_relative_transform
from tadataka.rigid_transform import transform_se3
from tadataka.decorator import allow_1d
from tadataka.projection import inv_pi, pi
from tadataka.pose import Pose
from rust_bindings import warp as _warp


def warp2d_(T10, xs0, depths0):
    return _warp.warp_vecs(T10, xs0, depths0)


def warp3d(T_w0, T_w1, P0):
    T_10 = calc_relative_transform(T_w1, T_w0)
    P1 = transform_se3(T_10, P0)
    return P1


def warp2d(T_wa, T_wb, xs, depths):
    PA = inv_pi(xs, depths)
    PB = warp3d(T_wa, T_wb, PA)
    return pi(PB)


class Warp3D(object):
    def __init__(self, pose_w0, pose_w1):
        assert(isinstance(pose_w0, Pose))
        assert(isinstance(pose_w1, Pose))
        self.T_w0 = pose_w0.T
        self.T_w1 = pose_w1.T

    @allow_1d(which_argument=1)
    def __call__(self, P):
        return warp3d(self.T_w0, self.T_w1, P)


def warp_depth(warp: Warp3D, xs0, depths0):
    P0 = inv_pi(xs0, depths0)
    P1 = warp(P0)
    xs1, depths1 = pi(P1), P1[:, 2]
    return xs1, depths1


class Warp2D(object):
    """Warp coordinate between image planes"""
    def __init__(self, camera_model0, camera_model1,
                 pose_w0: Pose, pose_w1: Pose):
        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1
        self.warp3d = Warp3D(pose_w0, pose_w1)

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


class LocalWarp2D(object):
    def __init__(self, camera_model0, camera_model1, pose10: Pose):
        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1
        self.T10 = pose10.T

    def __call__(self, us0, depths0):
        xs0 = self.camera_model0.normalize(us0)
        xs1, depths1 = warp2d_(self.T10, xs0, depths0)
        us1 = self.camera_model1.unnormalize(xs1)
        return us1, depths1
