import numpy as np

from tadataka.projection import pi
from tadataka.utils import is_in_image_range


def coordinates_along_line(start, step, disparities):
    return start + np.outer(disparities, step)


def normalize_length(v):
    return v / np.linalg.norm(v)


class EpipolarDirection(object):
    def __init__(self, t):
        self.x0 = pi(self.t)

    def __call__(self, x):
        return normalize_length(self.x0 - x)


def calc_coordinates(x_min, x_max, search_step):
    direction = normalize_length(x_max - x_min)
    N = np.linalg.norm(x_max - x_min) / search_step
    return coordinates_along_line(x_min, direction,
                                  search_step * np.arange(N))


class ReferenceCoordinates(object):
    def __init__(self, camera_model, image_shape, search_step):
        self.camera_model = camera_model
        self.image_shape = image_shape
        self.search_step = search_step

    def __call__(self, x_min, x_max):
        xs = calc_coordinates(x_min, x_max, self.search_step)
        us = self.camera_model.unnormalize(xs)
        mask = is_in_image_range(us, self.image_shape)
        return xs[mask], us[mask]
