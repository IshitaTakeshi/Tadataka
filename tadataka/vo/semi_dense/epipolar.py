import numpy as np
import numba

from tadataka.projection import pi
from tadataka.vector import normalize_length
from tadataka.utils import is_in_image_range


@numba.njit
def coordinates_along_line(start, step, disparities):
    return start + np.outer(disparities, step)


@numba.njit
def calc_coordinates(x_min, x_max, step_size):
    direction = normalize_length(x_max - x_min)
    N = np.linalg.norm(x_max - x_min) / step_size
    return coordinates_along_line(x_min, step_size * direction, np.arange(N))


@numba.njit
def reference_coordinates(x_range, step_size):
    x_min, x_max = x_range
    return calc_coordinates(x_min, x_max, step_size)


@numba.njit
def key_coordinates_(x, pi_t, step_size, sampling_steps):
    direction = normalize_length(x - pi_t)
    step = step_size * direction
    return coordinates_along_line(x, step, sampling_steps)


def key_coordinates(x, pi_t, step_size, sampling_steps=[-2, -1, 0, 1, 2]):
    return key_coordinates_(x, pi_t, step_size, np.array(sampling_steps))
