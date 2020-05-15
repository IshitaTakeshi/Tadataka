import numpy as np

from tadataka.projection import pi
from tadataka.vector import normalize_length
from tadataka.utils import is_in_image_range
from tadataka.matrix import inv_motion_matrix, get_translation
from tadataka.vo.semi_dense._epipolar import key_coordinates_, calc_coordinates
from tadataka.warp import warp2d_


EPSILON = 1e-16


def ref_coordinates(x_range, step_size):
    x_min, x_max = x_range
    return calc_coordinates(x_min, x_max, step_size)


sampling_steps = np.array([-2, -1, 0, 1, 2])


def key_epipolar_direction(t_rk, x_key):
    return x_key - pi(t_rk)


def key_coordinates(t_rk, x_key, step_size_key):
    return key_coordinates_(
            key_epipolar_direction(t_rk, x_key),
            x_key, step_size_key)


def ref_search_range(T_rk, x_key, depth_range):
    min_depth, max_depth = depth_range

    xs_key = np.vstack((x_key, x_key))
    depths_key = np.array([min_depth, max_depth])

    xs_ref = np.empty(xs_key.shape)
    depths_ref = np.empty(depths_key.shape)

    xs_ref, depths_ref = warp2d_(T_rk, xs_key, depths_key)

    x_min_ref, x_max_ref = xs_ref
    return x_min_ref, x_max_ref
