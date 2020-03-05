import numpy as np


EPSILON = 1e-16


def photometric_variance(gradient_along_epipolar_line, sigma_i):
    # Calculate photometric disparity error
    return 2 * (sigma_i * sigma_i) / gradient_along_epipolar_line


def geometric_variance(x, pi_t, image_gradient, sigma_l):
    direction = x - pi_t
    # Calculate geometric disparity error
    p = np.dot(direction, image_gradient)

    # normalization terms
    ng = np.dot(image_gradient, image_gradient)
    nl = np.dot(direction, direction)
    var = sigma_l * sigma_l

    return (var * ng * nl) / (p * p + EPSILON)


def calc_alpha_(x_key, x_ref_i, direction_i, ri, rz, ti, tz):
    x = np.append(x_key, 1)

    d = np.dot(rz, x) * ti - np.dot(ri, x) * tz
    n = x_ref_i * tz - ti

    return direction_i * d / (n * n)


def alpha_index(search_step):
    return np.argmax(np.abs(search_step))


def calc_alpha(x_key, x_ref, direction, R, t):
    index = alpha_index(direction)
    return calc_alpha_(x_key, x_ref[index], direction[index],
                       R[index], R[2], t[index], t[2])
