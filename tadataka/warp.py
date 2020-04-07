from numba import njit
from tadataka.rigid_transform import transform
from tadataka.decorator import allow_1d
from tadataka.projection import inv_pi, pi
from tadataka.rigid_transform import transform, inv_transform
from tadataka.pose import LocalPose, WorldPose


@njit
def warp3d(T0, T1, P):
    R0, t0 = T0
    R1, t1 = T1

    P = transform(R0, t0, P)  # camera 0 to world
    P = inv_transform(R1, t1, P)  # world to camera 1
    return P


@njit
def warp2d(T0, T1, xs, depths):
    P = inv_pi(xs, depths)
    Q = warp3d(T0, T1, P)
    return pi(Q)


class Warp3D(object):
    def __init__(self, pose0, pose1):
        assert(isinstance(pose0, WorldPose))
        assert(isinstance(pose1, WorldPose))
        self.T0 = pose0.R, pose0.t
        self.T1 = pose1.R, pose1.t

    @allow_1d(which_argument=1)
    def __call__(self, P):
        return warp3d(self.T0, self.T1, P)


def warp3d_(warp: Warp3D, xs0, depths0):
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

        xs1, depths1 = warp3d_(self.warp3d, xs0, depths0)

        us1 = self.camera_model1.unnormalize(xs1)
        return us1, depths1


def local_warp3d_(T10, xs0, depths0):
    R10, t10 = T10

    P0 = inv_pi(xs0, depths0)
    P1 = transform(R10, t10, P0)
    xs1, depths1 = pi(P1), P1[:, 2]
    return xs1, depths1


class LocalWarp2D(object):
    def __init__(self, camera_model0, camera_model1, pose10: LocalPose):
        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1
        self.T10 = pose10.R, pose10.t

    def __call__(self, us0, depths0):
        xs0 = self.camera_model0.normalize(us0)
        xs1, depths1 = local_warp3d_(self.T10, xs0, depths0)
        us1 = self.camera_model1.unnormalize(xs1)
        return us1, depths1
