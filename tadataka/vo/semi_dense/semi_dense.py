import numpy as np
from numba import njit

from tqdm import tqdm

from tadataka.coordinates import image_coordinates
from tadataka.vector import normalize_length
from tadataka.utils import is_in_image_range
from tadataka.matrix import to_homogeneous
from tadataka.camera.table import NoramlizationMapTable
from tadataka.projection import inv_pi, pi
from tadataka.rigid_transform import inv_transform
from tadataka.interpolation import interpolation1d_, interpolation2d_
from tadataka.triangulation import depth_from_triangulation
from tadataka.warp import warp2d, warp3d
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.vo.semi_dense.epipolar import (
    reference_coordinates, key_coordinates
)
from tadataka.vo.semi_dense.variance import (
    photometric_variance, geometric_variance, calc_alpha
)
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
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


def calc_depth_ref(T_key, T_ref, x_key, depth_key):
    p_key = inv_pi(x_key, depth_key)
    p_ref = warp3d(T_key, T_ref, p_key)
    return p_ref[2]


def step_size_ratio(T_key, T_ref, x_key, inv_depth_key):
    depth_key = invert_depth(inv_depth_key)
    depth_ref = calc_depth_ref(T_key, T_ref, x_key, depth_key)
    inv_depth_ref = invert_depth(depth_ref)
    return inv_depth_key / inv_depth_ref


class GradientImage(object):
    def __init__(self, image_grad_x, image_grad_y):
        self.grad_x = image_grad_x
        self.grad_y = image_grad_y

    def __call__(self, u_key):
        gx = interpolation1d_(self.grad_x, u_key)
        gy = interpolation1d_(self.grad_y, u_key)
        return np.array([gx, gy])


@njit
def depth_search_range(inv_depth_range):
    min_inv_depth, max_inv_depth = inv_depth_range
    min_depth = invert_depth(max_inv_depth)
    max_depth = invert_depth(min_inv_depth)
    return min_depth, max_depth


@njit
def epipolar_search_range_(T_key, T_ref, x_key, depth_range):
    min_depth, max_depth = depth_range

    x_ref_min = warp2d(T_key, T_ref, x_key, min_depth)
    x_ref_max = warp2d(T_key, T_ref, x_key, max_depth)

    return x_ref_min, x_ref_max


@njit
def epipolar_search_range(T_key, T_ref, x_key, inv_depth_range):
    depth_range = depth_search_range(inv_depth_range)
    return epipolar_search_range_(T_key, T_ref, x_key, depth_range)


class Uncertaintity(object):
    def __init__(self, sigma_i, sigma_l):
        self.sigma_l = sigma_l
        self.sigma_i = sigma_i

    def __call__(self, x_key, x_ref, x_range_ref, R, t,
                 image_grad, epipolar_gradient):
        geo_epsilon = 1e-4
        x_min_ref, x_max_ref = x_range_ref

        direction = normalize_length(x_max_ref - x_min_ref)
        alpha = calc_alpha(x_key, x_ref, direction, R, t)

        geo_var = geometric_variance(x_key - pi(t), image_grad,
                                     self.sigma_l, geo_epsilon)
        photo_var = photometric_variance(epipolar_gradient, self.sigma_i)
        return calc_observation_variance(alpha, geo_var, photo_var)


def _unnormalize_if_in_image(camera_model, xs, shape):
    us = camera_model.unnormalize(xs)
    mask = is_in_image_range(us, shape)
    return xs[mask], us[mask]


class InverseDepthEstimator(object):
    def __init__(self, keyframe, sigma_i, sigma_l, step_size_ref,
                 min_gradient):
        assert(np.ndim(keyframe.image) == 2)
        self.keyframe = keyframe

        self.T_key = (keyframe.pose.R, keyframe.pose.t)
        self.min_gradient = min_gradient

        image_key = keyframe.image

        self.uncertaintity = Uncertaintity(sigma_i, sigma_l)
        self.step_size_ref = step_size_ref
        self.image_grad = GradientImage(grad_x(image_key), grad_y(image_key))
        self.search_range = InverseDepthSearchRange(
            min_inv_depth=0.1, max_inv_depth=10.0
        )

    def __call__(self, refframe, u_key, prior_inv_depth, prior_variance):
        key = self.keyframe
        ref = refframe

        T_key = self.T_key  # key to world
        t_ref = ref.pose.t
        T_ref = (ref.pose.R, ref.pose.t)  # ref to world

        x_key = key.camera_model.normalize(u_key)

        ratio = step_size_ratio(T_key, T_ref, x_key, prior_inv_depth)
        step_size_key = ratio * self.step_size_ref

        # t_ref is the position of ref camera center in the world coordinate
        # inv_transform(*T_key, t_ref) brings the ref camera center onto
        # the key camera coordinate
        # pi project it onto the key image plane
        t_image_ref = pi(inv_transform(*T_key, t_ref))
        xs_key = key_coordinates(x_key, t_image_ref, step_size_key)
        us_key = key.camera_model.unnormalize(xs_key)

        if not is_in_image_range(us_key, key.image.shape).all():
            return prior_inv_depth, prior_variance, FLAG.KEY_OUT_OF_RANGE

        intensities_key = interpolation2d_(key.image, us_key)
        epipolar_gradient = intensity_gradient(intensities_key, step_size_key)
        if epipolar_gradient < self.min_gradient:
            return prior_inv_depth, prior_variance, FLAG.INSUFFICIENT_GRADIENT

        inv_depth_range = self.search_range(prior_inv_depth, prior_variance)
        x_range_ref = epipolar_search_range(T_key, T_ref, x_key, inv_depth_range)

        xs_ref = reference_coordinates(x_range_ref, self.step_size_ref)
        xs_ref, us_ref = _unnormalize_if_in_image(ref.camera_model, xs_ref,
                                                  ref.image.shape)

        if len(xs_ref) < len(xs_key):
            return prior_inv_depth, prior_variance, FLAG.EPIPOLAR_TOO_SHORT

        intensities_ref = interpolation2d_(ref.image, us_ref)
        argmin = search_intensities(intensities_key, intensities_ref)

        R_ref, t_ref = T_ref
        R_key, t_key = T_key

        # transformation from key camera coordinate to ref camera coordinate
        R = np.dot(R_ref.T, R_key)
        t = np.dot(R_ref.T, t_key - t_ref)

        depth_key = depth_from_triangulation(R, t, x_key, xs_ref[argmin])

        x_ref = warp2d(T_key, T_ref, x_key, invert_depth(prior_inv_depth))
        image_grad = self.image_grad(u_key)
        variance = self.uncertaintity(x_key, x_ref, x_range_ref, R, t,
                                      image_grad, epipolar_gradient)
        return invert_depth(depth_key), variance, FLAG.SUCCESS



class InverseDepthMapEstimator(object):
    def __init__(self, keyframe):
        self._estimator = InverseDepthEstimator(
            keyframe,
            sigma_l=0.005, sigma_i=0.04,
            step_size_ref=0.005, min_gradient=10.0
        )

    def __call__(self, refframe,
                 prior_inv_depth_map, prior_variance_map):
        image_shape = prior_inv_depth_map.shape

        inv_depth_map = np.zeros(image_shape)
        variance_map = np.zeros(image_shape)
        flag_map = np.zeros(image_shape)
        for u_key in tqdm(image_coordinates(image_shape)):
            x, y = u_key
            inv_depth, variance, flag = self._estimator(
                refframe, u_key,
                prior_inv_depth_map[y, x],
                prior_variance_map[y, x]
            )
            inv_depth_map[y, x] = inv_depth
            variance_map[y, x] = variance
            flag_map[y, x] = flag
        return inv_depth_map, variance_map, flag_map
