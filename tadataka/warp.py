from tadataka.rigid_transform import transform
from tadataka.decorator import allow_1d
from tadataka.projection import inv_pi, pi


class Warp3D(object):
    def __init__(self, pose0, pose1):
        camera0_to_world = pose0
        world_to_camera1 = pose1.to_local()

        self.R0, self.t0 = camera0_to_world.R, camera0_to_world.t
        self.R1, self.t1 = world_to_camera1.R, world_to_camera1.t

    @allow_1d(which_argument=1)
    def __call__(self, P):
        P = transform(self.R0, self.t0, P)
        P = transform(self.R1, self.t1, P)
        return P


def warp(warp3d, xs0, depths0):
    P0 = inv_pi(xs0, depths0)
    P1 = warp3d(P0)
    xs1, depths1 = pi(P1), P1[:, 2]
    return xs1, depths1


class Warp2D(object):
    def __init__(self, camera_model0, camera_model1, pose0, pose1):
        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1
        self.warp3d = Warp3D(pose0, pose1)

    def __call__(self, us0, depths0):
        xs0 = self.camera_model0.normalize(us0)

        xs1, depths1 = warp(self.warp3d, xs0, depths0)

        us1 = self.camera_model1.unnormalize(xs1)
        return us1, depths1
