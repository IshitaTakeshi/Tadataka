from numba import njit
from tadataka.rigid_transform import transform
from tadataka.decorator import allow_1d
from tadataka.projection import inv_pi, pi
from tadataka.rigid_transform import transform, inv_transform
from tadataka.pose import WorldPose


@njit
def warp3d(T0, T1, P):
    R0, t0 = T0
    R1, t1 = T1

    P = transform(R0, t0, P)  # camera 0 to world
    P = inv_transform(R1, t1, P)  # world to camera 1
    return P


@njit
def warp2d(T0, T1, x, depth):
    p = inv_pi(x, depth)
    q = warp3d(T0, T1, p)
    return pi(q)


class Warp3D(object):
    def __init__(self, pose0, pose1):
        assert(isinstance(pose0, WorldPose))
        assert(isinstance(pose1, WorldPose))
        self.T0 = pose0.R, pose0.t
        self.T1 = pose1.R, pose1.t

    @allow_1d(which_argument=1)
    def __call__(self, P):
        return warp3d(self.T0, self.T1, P)


def warp_(warp3d_, xs0, depths0):
    P0 = inv_pi(xs0, depths0)
    P1 = warp3d_(P0)
    xs1, depths1 = pi(P1), P1[:, 2]
    return xs1, depths1


class Warp2D(object):
    def __init__(self, camera_model0, camera_model1, pose0, pose1):
        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1
        self.warp3d = Warp3D(pose0, pose1)

    def __call__(self, us0, depths0):
        xs0 = self.camera_model0.normalize(us0)

        xs1, depths1 = warp_(self.warp3d, xs0, depths0)

        us1 = self.camera_model1.unnormalize(xs1)
        return us1, depths1
