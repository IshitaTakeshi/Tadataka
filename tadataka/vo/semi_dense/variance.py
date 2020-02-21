import numpy as np

from tadataka.vector import normalize_length
from tadataka.matrix import to_homogeneous


class PhotometricVariance(object):
    def __init__(self, sigma_i):
        self.sigma_i = sigma_i

    def __call__(self, gradient_along_epipolar_line):
        # Calculate photometric disparity error
        sigma_i = self.sigma_i
        return 2 * (sigma_i * sigma_i) / gradient_along_epipolar_line


class GeometricVariance(object):
    def __init__(self, epipolar_direction, sigma_l):
        self.epipolar_direction = epipolar_direction
        self.sigma_l = sigma_l

    def __call__(self, x, image_gradient):
        direction = self.epipolar_direction(x)
        # Calculate geometric disparity error
        p = np.dot(direction, image_gradient)

        # normalization terms
        ng = np.dot(image_gradient, image_gradient)
        nl = np.dot(direction, direction)
        var = self.sigma_l * self.sigma_l

        return (var * ng * nl) / (p * p)


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
