import numpy as np
from tadataka.matrix import get_rotation_translation
from tadataka.warp import warp2d_
from tadataka.numeric import safe_invert


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


def calc_alpha_(x_key, x_ref_i, direction_i, ri, rz, ti, tz):
    x = np.append(x_key, 1)

    d = np.dot(rz, x) * ti - np.dot(ri, x) * tz
    n = x_ref_i * tz - ti

    return direction_i * d / (n * n)


def alpha_index(search_step):
    return np.argmax(np.abs(search_step))


def calc_alpha(T_rk, x_key, direction, prior_inv_depth):
    x_ref, _ = warp2d_(T_rk, x_key, safe_invert(prior_inv_depth))
    R_rk, t_rk = get_rotation_translation(T_rk)
    index = alpha_index(direction)
    return calc_alpha_(x_key, x_ref[index], direction[index],
                       R_rk[index], R_rk[2], t_rk[index], t_rk[2])
