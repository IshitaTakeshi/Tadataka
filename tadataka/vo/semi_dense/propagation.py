import numpy as np

from tadataka.pose import WorldPose
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.coordinates import image_coordinates
from tadataka.utils import is_in_image_range
from tadataka.warp import Warp2D


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


def coordinates(warp10, depth_map0):
    us0 = image_coordinates(depth_map0.shape)
    depths0 = depth_map0.flatten()
    us1, depths1 = warp10(us0, depths0)

    mask = is_in_image_range(us1, depth_map0.shape)
    us0, depths0 = us0[mask], depths0[mask]
    us1, depths1 = us1[mask], depths1[mask]

    return us0, us1, depths0, depths1


def propagate_(warp10, depth_map0, variance_map0, uncertaintity_bias=0.01):
    assert(depth_map0.shape == variance_map0.shape)
    shape = depth_map0.shape

    us0, us1, depths0, depths1 = coordinates(warp10, depth_map0)

    us1 = np.round(us1).astype(np.int64)

    variances0 = get(variance_map0, us0)
    variances1 = propagate_variance(invert_depth(depths0),
                                    invert_depth(depths1),
                                    variances0, uncertaintity_bias)

    depth_map1 = substitute(np.ones(shape), us1, depths1)
    variance_map1 = substitute(np.ones(shape), us1, variances1)
    return depth_map1, variance_map1


def propagate(camera_model0, camera_model1, warp10,
              inv_depth_map0, variance_map0):
    depth_map0 = invert_depth(inv_depth_map0)

    depth_map1, variance_map1 = propagate_(warp10, depth_map0, variance_map0)

    return invert_depth(depth_map1), variance_map1


def detect_intensity_change_(warp10, image0, image1, depth_map0,
                             threshold):
    from tadataka.interpolation import interpolation
    us0, us1, depths0, depths1 = coordinates(warp10, depth_map0)

    intensities0 = get(image0, us0)
    intensities1 = interpolation(image1, us1)
    mask = (intensities0 - intensities1) > threshold
    return substitute(np.zeros(image0.shape, dtype=np.bool), us0, mask)


def detect_intensity_change(warp10, image0, image1, inv_depth_map0,
                            threshold=0.2):
    assert(image0.dtype == np.float64)
    assert(image1.dtype == np.float64)
    return detect_intensity_change_(warp10, image0, image1,
                                    invert_depth(inv_depth_map0), threshold)


def warp_image(warp10, depth_map0, image0, default_value=0.0):
    us0, us1, depths0, depths1 = coordinates(warp10, depth_map0)
    us1 = np.round(us1).astype(np.int64)
    default = np.full(depth_map0.shape, default_value)
    return substitute(default, us0, get(image0, us0))
