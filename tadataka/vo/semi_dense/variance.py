import numpy as np


def photometric_variance(gradient_along_epipolar_line, sigma_i):
    # Calculate photometric disparity error
    return 2 * (sigma_i * sigma_i) / gradient_along_epipolar_line


def geometric_variance(epipolar_direction, image_gradient, sigma_l, epsilon):
    ng = np.dot(image_gradient, image_gradient)
    nl = np.dot(epipolar_direction, epipolar_direction)
    normalizer = ng * nl

    if normalizer < epsilon:
        return (sigma_l * sigma_l) / epsilon

    p = np.dot(epipolar_direction, image_gradient)
    product = (p * p) / normalizer

    if product < epsilon:
        return (sigma_l * sigma_l) / epsilon

    return (sigma_l * sigma_l) / product


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
