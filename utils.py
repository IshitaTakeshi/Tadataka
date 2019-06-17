from autograd import numpy as np


def from_2d(x):
    return x.flatten()


def to_2d(x):
    return x.reshape(-1, 2)


def is_in_image_range(points, image_shape):
    height, width = image_shape
    xs, ys = points[:, 0], points[:, 1]
    mask_x = np.logical_and(0 <= xs, xs < width)
    mask_y = np.logical_and(0 <= ys, ys < width)
    return np.logical_and(mask_x, mask_y)
