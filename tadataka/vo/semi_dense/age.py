import numpy as np

from tadataka.warp import warp2d
from tadataka.utils import is_in_image_range
from tadataka.coordinates import image_coordinates
from tadataka.vo.semi_dense.common import invert_depth


def increment_(age0, us0, us1):
    xs0, ys0 = us0[:, 0], us0[:, 1]
    xs1, ys1 = us1[:, 0], us1[:, 1]

    age1 = np.zeros(age0.shape, dtype=age0.dtype)
    age1[ys1, xs1] = age0[ys0, xs0] + 1
    return age1


def increment_age(age0, inv_depth_map0,
                  camera_model0, camera_model1, T0, T1):
    # T0 = (R0, t0), T1 = (R1, t1)
    assert(age0.shape == inv_depth_map0.shape)

    depth_map0 = invert_depth(inv_depth_map0)
    image_shape = age0.shape
    depths0 = depth_map0.flatten()

    us0 = image_coordinates(image_shape)
    xs0 = camera_model0.normalize(us0)
    xs1 = warp2d(T0, T1, xs0, depths0)
    us1 = camera_model1.unnormalize(xs1)

    us1 = np.round(us1).astype(np.int64)
    mask = is_in_image_range(us1, image_shape)
    return increment_(age0, us0[mask], us1[mask])
