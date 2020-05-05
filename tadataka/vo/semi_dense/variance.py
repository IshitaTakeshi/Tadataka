import numpy as np
from tadataka.matrix import get_rotation_translation
from tadataka.warp import warp2d_
from tadataka.numeric import safe_invert
from tadataka.vo.semi_dense import _variance


def calc_observation_variance(alpha, geo_variance, photo_variance):
    return alpha * alpha * (geo_variance + photo_variance)


def photometric_variance(gradient_along_epipolar_line, sigma_i):
    # Calculate photometric disparity error
    return 2 * (sigma_i * sigma_i) / gradient_along_epipolar_line


def geometric_variance(epipolar_direction, image_gradient, sigma_l, epsilon):
    # we assume epipolar_direction and image_gradient are not normalized
    ng = np.dot(image_gradient, image_gradient)
    nl = np.dot(epipolar_direction, epipolar_direction)
    normalizer = ng * nl
    sl2 = (sigma_l * sigma_l)

    if normalizer < epsilon:
        return sl2 / epsilon

    p = np.dot(epipolar_direction, image_gradient)
    product = (p * p) / normalizer

    if product < epsilon:
        return sl2 / epsilon

    return sl2 / product


def calc_alpha(T_rk, x_key, direction, prior_inv_depth):
    return _variance.calc_alpha(T_rk, x_key, direction,
                                safe_invert(prior_inv_depth))

