import numpy as np
from numba import njit

from tadataka.projection import pi
from tadataka.vector import normalize_length
from tadataka.utils import is_in_image_range


@njit
def coordinates_along_line(start, step, disparities):
    return start + np.outer(disparities, step)


@njit
def calc_coordinates(x_min, x_max, step_size):
    d = x_max - x_min
    norm = np.linalg.norm(d)
    direction = d / norm
    N = norm // step_size
    return coordinates_along_line(x_min, step_size * direction, np.arange(N))


@njit
def reference_coordinates(x_range, step_size):
    x_min, x_max = x_range
    return calc_coordinates(x_min, x_max, step_size)


@njit
def key_coordinates_(x, pi_t, step_size, sampling_steps):
    direction = normalize_length(x - pi_t)
    step = step_size * direction
    return coordinates_along_line(x, step, sampling_steps)


@njit
def key_coordinates(x, pi_t, step_size):
    sampling_steps = np.array([-2, -1, 0, 1, 2])
    return key_coordinates_(x, pi_t, step_size, sampling_steps)
