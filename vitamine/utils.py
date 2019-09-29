from autograd import numpy as np


def random_binary(size):
    return np.random.randint(0, 2, size, dtype=np.bool)


def indices_other_than(size, indices):
    """
    size: size of the array you want to get elements from
    example:
    >>> indices_other_than(8, [1, 2, 3])
    [0, 4, 5, 6, 7]
    """
    return np.setxor1d(indices, np.arange(size))


def round_int(X):
    return np.round(X, 0).astype(np.int64)


def is_in_image_range(keypoints, image_shape):
    height, width = image_shape
    xs, ys = keypoints[:, 0], keypoints[:, 1]
    mask_x = np.logical_and(0 <= xs, xs < width)
    mask_y = np.logical_and(0 <= ys, ys < height)
    mask = np.logical_and(mask_x, mask_y)
    return mask


def radian_to_degree(radian):
    return radian / np.pi * 180


def add_noise(descriptors, indices):
    descriptors = np.copy(descriptors)
    descriptors[indices] = random_binary((len(indices), descriptors.shape[1]))
    return descriptors


def break_other_than(descriptors, indices):
    indices_to_break = np.setxor1d(np.arange(len(descriptors)), indices)
    return add_noise(descriptors, indices_to_break)
