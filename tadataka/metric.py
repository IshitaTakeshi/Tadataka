import numpy as np
from tadataka.interpolation import interpolation_
from tadataka.warp import LocalWarp2D
from tadataka.coordinates import image_coordinates
from tadataka.utils import is_in_image_range


def calc_error_(v1, v2):
    return np.mean(np.power(v1 - v2, 2))


def photometric_error(warp, gray_image0, depth_map0, gray_image1):
    # TODO change the argument order
    #    gray_image0, depth_map0, gray_image1
    # -> gray_image0, gray_image1, depth_map0
    # assert(isinstance(warp, LocalWarp2D))

    us0 = image_coordinates(depth_map0.shape)
    us1, depths1 = warp(us0, depth_map0.flatten())

    mask = is_in_image_range(us1, depth_map0.shape)

    i0 = gray_image0[us0[mask, 1], us0[mask, 0]]
    i1 = interpolation_(gray_image1, us1[mask])

    return calc_error_(i0, i1)


class PhotometricError(object):
    def __init__(self, camera_model0, camera_model1, I0, D0, I1):
        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1
        self.I0, self.D0, self.I1 = I0, D0, I1

    def __call__(self, pose10):
        # warp points in t0 coordinate onto the t1 coordinate
        warp10 = LocalWarp2D(self.camera_model0, self.camera_model1, pose10)
        return photometric_error(warp10, self.I0, self.D0, self.I1)
