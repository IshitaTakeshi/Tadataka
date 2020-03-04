import numpy as np
import numba

from tadataka.coordinates import image_coordinates
from tadataka.utils import is_in_image_range
from tadataka.matrix import to_homogeneous
from tadataka.projection import pi
from tadataka.interpolation import interpolation
from tadataka.triangulation import DepthFromTriangulation
from tadataka.rigid_transform import transform
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.vo.semi_dense.epipolar import (
    reference_coordinates, key_coordinates
)
from tadataka.vo.semi_dense.variance import (
    photometric_variance, geometric_variance, Alpha
)
from tadataka.vo.semi_dense.intensities import search_intensities
from tadataka.gradient import grad_x, grad_y


def intensity_gradient(intensities, interval):
    return np.linalg.norm(intensities[1:] - intensities[:-1]) / interval


def calc_observation_variance(alpha, geo_variance, photo_variance):
    return alpha * alpha * (geo_variance + photo_variance)


class InverseDepthSearchRange(object):
    def __init__(self, min_inv_depth, max_inv_depth, factor=2.0):
        assert(min_inv_depth > 0.0)
        assert(max_inv_depth > min_inv_depth)
        self.factor = factor
        self.min_inv_depth = min_inv_depth
        self.max_inv_depth = max_inv_depth

    def __call__(self, inv_depth, variance):
        assert(variance >= 0.0)
        L = max(inv_depth - self.factor * variance, self.min_inv_depth)
        U = min(inv_depth + self.factor * variance, self.max_inv_depth)
        return L, U


def calc_depth_ref(x_key, depth_key, R, t):
    q = transform(R, t, depth_key * to_homogeneous(x_key))
    return q[2]


def step_size_ratio(x_key, inv_depth_key, R, t):
    depth_key = invert_depth(inv_depth_key)
    depth_ref = calc_depth_ref(x_key, depth_key, R, t)
    inv_depth_ref = invert_depth(depth_ref)
    return inv_depth_key / inv_depth_ref


class GradientImage(object):
    def __init__(self, image_grad_x, image_grad_y):
        self.grad_x = image_grad_x
        self.grad_y = image_grad_y

    def __call__(self, u_key):
        gx = interpolation(self.grad_x, u_key)
        gy = interpolation(self.grad_y, u_key)
        return np.array([gx, gy])


def depth_search_range(inv_depth_range):
    min_inv_depth, max_inv_depth = inv_depth_range
    min_depth = invert_depth(max_inv_depth)
    max_depth = invert_depth(min_inv_depth)
    return min_depth, max_depth


def epipolar_search_range(x_key, depth_range, R, t):
    min_depth, max_depth = depth_range
    x_ref_min = pi(transform(R, t, to_homogeneous(x_key) * min_depth))
    x_ref_max = pi(transform(R, t, to_homogeneous(x_key) * max_depth))
    return x_ref_min, x_ref_max


class InverseDepthEstimator(object):
    def __init__(self, image_key, camera_model_key,
                 sigma_i, sigma_l, step_size_ref,
                 min_gradient):
        assert(np.ndim(image_key) == 2)

        self.sigma_l = sigma_l
        self.sigma_i = sigma_i
        self.image_key = image_key
        self.camera_model_key = camera_model_key
        self.min_gradient = min_gradient
        # -1 for bilinear interpolation
        height, width = image_key.shape
        self.coordinate_range = (height-1, width-1)

        self.step_size_ref = step_size_ref
        self.image_grad = GradientImage(grad_x(image_key), grad_y(image_key))
        self.search_range = InverseDepthSearchRange(
            min_inv_depth=0.05, max_inv_depth=10.0
        )

    def __call__(self, pose_key_to_ref, camera_model_ref, image_ref, u_key,
                 prior_inv_depth, prior_variance):
        R, t = pose_key_to_ref.R, pose_key_to_ref.t

        # image of reference camera center on the keyframe
        pi_t = pi(t)

        x_key = self.camera_model_key.normalize(u_key)
        ratio = step_size_ratio(x_key, prior_inv_depth, R, t)
        step_size_key = ratio * self.step_size_ref
        xs_key = key_coordinates(x_key, pi_t, step_size_key)
        us_key = self.camera_model_key.unnormalize(xs_key)

        if not is_in_image_range(us_key, self.coordinate_range).all():
            return prior_inv_depth, prior_variance

        depth_range = depth_search_range(
            self.search_range(prior_inv_depth, prior_variance)
        )
        x_range_ref = epipolar_search_range(x_key, depth_range, R, t)

        xs_ref = reference_coordinates(x_range_ref, self.step_size_ref)
        us_ref = camera_model_ref.unnormalize(xs_ref)
        mask = is_in_image_range(us_ref, self.coordinate_range)
        xs_ref, us_ref = xs_ref[mask], us_ref[mask]

        if len(xs_ref) < len(xs_key):
            return prior_inv_depth, prior_variance

        intensities_key = interpolation(self.image_key, us_key)
        epipolar_gradient = intensity_gradient(intensities_key,
                                               np.abs(step_size_key))
        if epipolar_gradient < self.min_gradient:
            return prior_inv_depth, prior_variance

        intensities_ref = interpolation(image_ref, us_ref)
        argmin = search_intensities(intensities_key, intensities_ref)
        x_ref = xs_ref[argmin]

        u_ref = camera_model_ref.unnormalize(x_ref)

        key_depth = DepthFromTriangulation(pose_key_to_ref)(x_key, x_ref)

        alpha = Alpha(R, t)(x_key, x_ref, x_range_ref)

        image_grad = self.image_grad(u_key)
        geo_var = geometric_variance(x_key, pi_t, image_grad, self.sigma_l)
        photo_var = photometric_variance(epipolar_gradient, self.sigma_i)
        variance = calc_observation_variance(alpha, geo_var, photo_var)

        return invert_depth(key_depth), variance


def estimate_inverse_depth_map(estimator,
                               prior_inv_depth_map, prior_variance_map):
    assert(prior_inv_depth_map.shape == prior_variance_map.shape)
    search_range = InverseDepthSearchRange(min_inv_depth=1e-16,
                                           max_inv_depth=20.0)

    image_shape = prior_inv_depth_map.shape
    inv_depth_map = np.empty(image_shape)
    variance_map = np.empty(image_shape)
    for u_key in image_coordinates(image_shape):
        x, y = u_key
        prior_inv_depth = prior_inv_depth_map[y, x]
        prior_variance = prior_variance_map[y, x]

        inv_depth_range = search_range(prior_inv_depth, prior_variance)
        inv_depth, variance = estimator(u_key, prior_inv_depth, inv_depth_range)

        inv_depth_map[y, x] = inv_depth
        variance_map[y, x] = variance
    return inv_depth_map, variance_map
