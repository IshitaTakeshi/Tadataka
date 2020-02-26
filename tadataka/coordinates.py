import numpy as np

from tadataka.rigid_transform import rotate_each
from tadataka.so3 import exp_so3


def image_coordinates(image_shape):
    height, width = image_shape[0:2]
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    return np.column_stack((xs.flatten(), ys.flatten()))


def yx_to_xy(coordinates):
    return coordinates[:, [1, 0]]


def xy_to_yx(coordinates):
    # this is identical to 'yx_to_xy' but I prefer to name expilictly
    return yx_to_xy(coordinates)
