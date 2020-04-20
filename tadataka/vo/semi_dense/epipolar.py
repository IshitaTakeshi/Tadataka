import numpy as np

from tadataka.projection import pi
from tadataka.vector import normalize_length
from tadataka.utils import is_in_image_range
from tadataka.matrix import inv_motion_matrix, get_translation


def coordinates_along_line(start, step, disparities):
    return start + np.outer(disparities, step)


def calc_coordinates(x_min, x_max, step_size):
    d = x_max - x_min
    norm = np.linalg.norm(d)
    direction = d / norm
    N = norm // step_size
    return coordinates_along_line(x_min, step_size * direction, np.arange(N))


def reference_coordinates(x_range, step_size):
    x_min, x_max = x_range
    return calc_coordinates(x_min, x_max, step_size)


sampling_steps = np.array([-2, -1, 0, 1, 2])


def key_coordinates(epipolar_direction, x_key, step_size):
    direction = normalize_length(epipolar_direction)
    step = step_size * direction
    return coordinates_along_line(x_key, step, sampling_steps)


def key_epipolar_direction(t_rk, x_key):
    return x_key - pi(t_rk)
