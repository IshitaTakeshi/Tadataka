import numpy as np

from tadataka.pose import WorldPose
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.coordinates import image_coordinates
from tadataka.utils import is_in_image_range


def propagate_variance(inv_depths0, inv_depths1, variances0, uncertaintity):
    # we assume few rotation change between timestamp 0 and timestamp 1
    # TODO accept the case rotations significantly change between 0 and 1
    ratio = inv_depths1 / inv_depths0
    variances1 = np.power(ratio, 4) * variances0 + uncertaintity
    return variances1


def substitute(array2d, us, values):
    assert(us.shape[0] == values.shape[0])
    xs, ys = us[:, 0], us[:, 1]
    array2d[ys, xs] = values
    return array2d


def get(array2d, us):
    xs, ys = us[:, 0], us[:, 1]
    return array2d[ys, xs]


def coordinates(warp01, depth_map0):
    image_shape = depth_map0.shape

    us0 = image_coordinates(image_shape)
    depths0 = get(depth_map0, us0)

    us1, depths1 = warp01(us0, depths0)
    us1 = np.round(us1).astype(np.int64)

    mask = is_in_image_range(us1, image_shape)
    return us0[mask], us1[mask], depths0[mask], depths1[mask]
