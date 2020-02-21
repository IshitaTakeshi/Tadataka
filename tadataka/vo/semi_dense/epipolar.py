import numpy as np

from tadataka.projection import pi
from tadataka.utils import is_in_image_range


def coordinates_along_line(start, step, disparities):
    return start + np.outer(disparities, step)


def normalize_length(v):
    return v / np.linalg.norm(v)


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
    def __init__(self, camera_model, image_shape, step_size):
        self.camera_model = camera_model
        self.image_shape = image_shape
        self.step_size = step_size

    def __call__(self, x_range):
        x_min, x_max = x_range
        xs = calc_coordinates(x_min, x_max, self.step_size)
        us = self.camera_model.unnormalize(xs)
        mask = is_in_image_range(us, self.image_shape)
        return xs[mask], us[mask]


class NormalizedKeyCoordinates(object):
    def __init__(self, epipolar_direction, sampling_steps):
        self.epipolar_direction = epipolar_direction
        self.sampling_steps = sampling_steps

    def __call__(self, x, step_size):
        direction = normalize_length(self.epipolar_direction(x))
        step = step_size * direction
        return coordinates_along_line(x, step, self.sampling_steps)


class KeyCoordinates(object):
    def __init__(self, camera_model, epipolar_direction,
                 sampling_steps=[-2, -1, 0, 1, 2]):
        self.camera_model = camera_model
        self.normalized = NormalizedKeyCoordinates(epipolar_direction,
                                                   sampling_steps)

    def __call__(self, x, step_size):
        xs = self.normalized(x, step_size)
        return self.camera_model.unnormalize(xs)
