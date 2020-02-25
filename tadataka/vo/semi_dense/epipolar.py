import numpy as np

from tadataka.projection import pi
from tadataka.vector import normalize_length
from tadataka.utils import is_in_image_range


def coordinates_along_line(start, step, disparities):
    return start + np.outer(disparities, step)


class EpipolarDirection(object):
    def __init__(self, t):
        self.x0 = pi(t)

    def __call__(self, x):
        return x - self.x0


def calc_coordinates(x_min, x_max, step_size):
    direction = normalize_length(x_max - x_min)
    N = np.linalg.norm(x_max - x_min) / step_size
    return coordinates_along_line(x_min, step_size * direction, np.arange(N))


class ReferenceCoordinates(object):
    def __init__(self, step_size):
        self.step_size = step_size

    def __call__(self, x_range):
        x_min, x_max = x_range
        return calc_coordinates(x_min, x_max, self.step_size)


class KeyCoordinates(object):
    def __init__(self, epipolar_direction, sampling_steps=[-2, -1, 0, 1, 2]):
        self.epipolar_direction = epipolar_direction
        self.sampling_steps = sampling_steps

    def __call__(self, x, step_size):
        direction = normalize_length(self.epipolar_direction(x))
        step = step_size * direction
        return coordinates_along_line(x, step, self.sampling_steps)
