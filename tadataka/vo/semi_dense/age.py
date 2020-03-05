import numpy as np

from tadataka.utils import is_in_image_range
from tadataka.coordinates import image_coordinates


def increment_(age0, us0, us1):
    xs0, ys0 = us0[:, 0], us0[:, 1]
    xs1, ys1 = us1[:, 0], us1[:, 1]

    age1 = np.zeros(age0.shape, dtype=age0.dtype)
    age1[ys1, xs1] = age0[ys0, xs0] + 1
    return age1


def increment_age(age0, depth_map0, warp01):
    assert(age0.shape == depth_map0.shape)
    image_shape = age0.shape
    depths0 = depth_map0.flatten()
    us0 = image_coordinates(image_shape)
    us1 = np.round(warp01(us0, depths0)).astype(np.int64)
    mask = is_in_image_range(us1, image_shape)
    return increment_(age0, us0[mask], us1[mask])
