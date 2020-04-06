import numpy as np

from tadataka.vo.semi_dense.stat import are_statically_same
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.vo.semi_dense.coordinates import substitute, get, coordinates


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

    if invert_depth(inv_depth_a) < invert_depth(inv_depth_b):
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


def propagate(warp10, inv_depth_map0, variance_map0,
              default_inv_depth=1.0, default_variance=1.0,
              uncertaintity_bias=0.01):
    assert(inv_depth_map0.shape == variance_map0.shape)
    shape = inv_depth_map0.shape

    us0, us1, inv_depths0, inv_depths1 = coordinates(warp10, inv_depth_map0)
    variances1 = propagate_variance(inv_depths0, inv_depths1,
                                    get(variance_map0, us0),
                                    uncertaintity_bias)

    inv_depth_map1, variance_map1 = substitute_(us1.astype(np.int64),
                                                inv_depths1, variances1, shape)
    inv_depth_map1[np.isnan(inv_depth_map1)] = default_inv_depth
    variance_map1[np.isnan(variance_map1)] = default_variance
    return inv_depth_map1, variance_map1


from tadataka.interpolation import interpolation
def detect_intensity_change(warp10, image0, image1, inv_depth_map0,
                            threshold=0.2):
    assert(image0.dtype == np.float64)
    assert(image1.dtype == np.float64)
    us0, us1, _, _ = coordinates(warp10, inv_depth_map0)

    intensities0 = get(image0, us0)
    intensities1 = interpolation(image1, us1)
    mask = (intensities0 - intensities1) > threshold
    return substitute(np.zeros(image0.shape, dtype=np.bool), us0, mask)
