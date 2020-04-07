import numpy as np
from tadataka.interpolation import interpolation2d_
from tadataka.warp import LocalWarp2D
from tadataka.coordinates import image_coordinates
from tadataka.utils import is_in_image_range


def calc_error_(v1, v2):
    return np.mean(np.power(v1 - v2, 2))


def photometric_error(warp: LocalWarp2D, gray_image0, depth_map0, gray_image1):
    # TODO change the argument order
    #    gray_image0, depth_map0, gray_image1
    # -> gray_image0, gray_image1, depth_map0

    us0 = image_coordinates(depth_map0.shape)
    us1, depths1 = warp(us0, depth_map0.flatten())

    mask = is_in_image_range(us1, depth_map0.shape)

    i0 = gray_image0[us0[mask, 1], us0[mask, 0]]
    i1 = interpolation2d_(gray_image1, us1[mask])

    return calc_error_(i0, i1)
