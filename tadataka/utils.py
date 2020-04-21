import numpy as np

from tadataka.decorator import allow_1d


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


def merge_dicts(*dicts):
    merged = dict()
    for d in dicts:
        merged.update(d)
    return merged


def round_int(X):
    return np.round(X, 0).astype(np.int64)


def value_list(dict_, keys):
    return [dict_[k] for k in keys]


def _is_in_image_range(keypoints, image_shape):
    height, width = image_shape
    xs, ys = keypoints[:, 0], keypoints[:, 1]
    # using this form to accept float coordinates
    # if width is 200, [0.0, 199.0] is accepted
    mask_x = np.logical_and(0 <= xs, xs <= width-1)
    mask_y = np.logical_and(0 <= ys, ys <= height-1)
    return np.logical_and(mask_x, mask_y)


# TODO move this function to 'range.py' or some file of more explicit name
@allow_1d(which_argument=0)
def is_in_image_range(keypoints, image_shape):
    """
    Accept coordinates in range x <- [0, width-1], y <- [0, height-1]
    """

    # assert(isinstance(image_shape, tuple) or isinstance(image_shape, list))

    return _is_in_image_range(keypoints, image_shape[0:2])


def radian_to_degree(radian):
    return radian / np.pi * 180


def add_noise(descriptors, indices):
    descriptors = np.copy(descriptors)
    descriptors[indices] = random_binary((len(indices), descriptors.shape[1]))
    return descriptors


def break_other_than(descriptors, indices):
    indices_to_break = np.setxor1d(np.arange(len(descriptors)), indices)
    return add_noise(descriptors, indices_to_break)
