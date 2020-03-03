import numpy as np

from tadataka.utils import is_in_image_range
from tadataka.coordinates import image_coordinates


def increment_age(age0, warp01, depth_map0):
    image_shape = depth_map0.shape

    us0 = image_coordinates(image_shape)
    depths0 = depth_map0.flatten()
    us1 = warp01(us0, depths0)
    mask = is_in_image_range(us1, image_shape)

    xs0, ys0 = us0[mask, 0], us0[mask, 1]
    xs1, ys1 = us1[mask, 0], us1[mask, 1]

    age1 = np.zeros(image_shape, dtype=age0.dtype)
    age1[ys1, xs1] = age0[ys0, xs0] + 1
    return age1
