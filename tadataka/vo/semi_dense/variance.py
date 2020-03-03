import numpy as np

from tadataka.vector import normalize_length
from tadataka.matrix import to_homogeneous


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


def calc_alphas(x_key, x_ref, epipolar_direction_ref, R, t):
    rot_x = np.dot(R, to_homogeneous(x_key))

    d = rot_x[2] * t[0:2] - rot_x[0:2] * t[2]
    n = x_ref[0:2] * t[2] - t[0:2]

    return epipolar_direction_ref * d / (n * n)


def alpha_index(search_step):
    return np.argmax(np.abs(search_step))


class Alpha(object):
    def __init__(self, R, t):
        self.R, self.t = R, t

    def __call__(self, x_key, x_ref, x_range_ref):
        x_min_ref, x_max_ref = x_range_ref
        direction = normalize_length(x_max_ref - x_min_ref)
        alphas = calc_alphas(x_key, x_ref, direction, self.R, self.t)
        index = alpha_index(direction)
        return alphas[index]
