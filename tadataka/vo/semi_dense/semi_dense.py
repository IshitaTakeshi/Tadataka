import numpy as np
import numba

from tadataka.utils import is_in_image_range
from tadataka.matrix import to_homogeneous
from tadataka.projection import pi
from tadataka.interpolation import interpolation
from tadataka.triangulation import DepthFromTriangulation
from tadataka.pose import Pose


def inverse_projection(camera_model, x, depth):
    return depth * to_homogeneous(x)
def intensity_gradient(intensities, interval):
    return np.linalg.norm(intensities[1:] - intensities[:-1]) / interval


def calc_inv_depths(ref_coordinate, key_coordinate,
                    R_key_to_ref, t_key_to_ref):
    rot_inv_x = np.dot(R_key_to_ref, x_key)

    n = x_ref[0:2] * t_key_to_ref - t_key_to_ref[0:2]
    d = rot_inv_x[0:2] - ref_coordinate * rot_inv_x[2]
    return d / n


def depth_coordinate(search_step):
    return 0 if np.abs(search_step[0]) > np.abs(search_step[1]) else 1


def search_intensities(intensities_ref, intensities_key, error_func):
    errors = convolve(intensities_ref, intensities_key, error_func)
    return np.argmin(errors)


class InsufficientCoordinatesError(Exception):
    pass


def error_if_insufficient_coordinates(mask, min_coordinates):
    n_in_image = np.sum(mask)
    if n_in_image < min_coordinates:
        raise InsufficientCoordinatesError(
            "Insufficient number of coordinates "
            "sampled from the epipolar line. "
            "Required {}, but found {}.".format(min_coordinates, n_in_image)
        )


def update(d1, d2, var1, var2):
    d = (d1 * var2 + d2 * var1) / (var1 + var2)
    var = (var1 * var2) / (var1 + var2)
    return d, var
