import numpy as np

from tadataka.coordinates import substitute, get
from tadataka.vo.semi_dense.stat import are_statically_same
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.numeric import safe_invert
from tadataka.vo.semi_dense.coordinates import warp_coordinates


def propagate_variance(inv_depths0, inv_depths1, variances0, uncertaintity):
    # we assume few rotation change between timestamp 0 and timestamp 1
    # TODO accept the case rotations significantly change between 0 and 1
    ratio = inv_depths1 / inv_depths0
    variances1 = np.power(ratio, 4) * variances0 + uncertaintity
    return variances1


def handle_collision_(inv_depth_a, inv_depth_b, variance_a, variance_b):
    if are_statically_same(inv_depth_a, inv_depth_b,
                           variance_a, variance_b, factor=2.0):
        return fusion(inv_depth_a, inv_depth_b, variance_a, variance_b)

    # b is hidden by a
    if safe_invert(inv_depth_a) < safe_invert(inv_depth_b):
        return (inv_depth_a, variance_a)
    else:
        return (inv_depth_b, variance_b)


def substitute_(us, inv_depths, variances, shape):
    inv_depth_map = np.full(shape, np.nan)
    variance_map = np.full(shape, np.nan)
    for i in range(us.shape[0]):
        x, y = us[i]

        if np.isnan(inv_depth_map[y, x]):
            inv_depth_map[y, x] = inv_depths[i]
            variance_map[y, x] = variances[i]
            continue

        inv_depth_map[y, x], variance_map[y, x] = handle_collision_(
            inv_depth_map[y, x], inv_depths[i],
            variance_map[y, x], variances[i]
        )
    return inv_depth_map, variance_map


def substitute(us, inv_depths, variances, shape,
               default_inv_depth, default_variance):
    inv_depth_map, variance_map = substitute_(us.astype(np.int64),
                                              inv_depths, variances, shape)
    inv_depth_map[np.isnan(inv_depth_map)] = default_inv_depth
    variance_map[np.isnan(variance_map)] = default_variance
    return inv_depth_map, variance_map


def propagation(inv_depth_map0, variance_map0,
                default_inv_depth=1.0, default_variance=1.0,
                uncertaintity_bias=1.0):
    us0, us1, inv_depths0, inv_depths1 = warp_coordinates(
        warp10, inv_depth_map0
    )
    variances1 = propagate_variance(inv_depths0, inv_depths1,
                                    get(variance_map0, us0),
                                    uncertaintity_bias)
    return substitute(us1, inv_depths1, variances1, inv_depth_map0.shape,
                      default_inv_depth, default_variance)
