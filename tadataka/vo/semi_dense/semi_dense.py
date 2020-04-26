from warnings import warn

import numpy as np

from tqdm import tqdm

from tadataka.coordinates import image_coordinates
from tadataka.interpolation import interpolation
from tadataka.gradient import grad_x, grad_y
from tadataka.matrix import to_homogeneous, inv_motion_matrix, get_translation
from tadataka.rigid_transform import inv_transform
from tadataka.projection import inv_pi, pi
from tadataka.utils import is_in_image_range
from tadataka.vector import normalize_length
from tadataka.numeric import safe_invert
from tadataka.vo.semi_dense.depth import (
    calc_ref_inv_depth, calc_key_depth,
    InvDepthSearchRange, depth_search_range
)
from tadataka.vo.semi_dense.epipolar import (
    key_coordinates, key_epipolar_direction,
    ref_coordinates, ref_search_range
)
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tadataka.vo.semi_dense.gradient import GradientImage
from tadataka.vo.semi_dense._gradient import calc_gradient_norm
from tadataka.vo.semi_dense.variance import (
    calc_alpha, calc_observation_variance,
    photometric_variance, geometric_variance
)
from tadataka.vo.semi_dense.hypothesis import Hypothesis
from tadataka.vo.semi_dense._intensities import search_intensities
from tadataka.warp import warp2d, warp3d


def all_points_in_image(us, image_shape):
    return is_in_image_range(us, image_shape).all()


class InvDepthEstimator(object):
    def __init__(self, camera_model_key, image_key,
                 inv_depth_search_range, sigma_i, sigma_l,
                 step_size_ref, min_gradient):
        assert(np.ndim(image_key) == 2)
        assert(step_size_ref > 0)
        self.sigma_i = sigma_i
        self.sigma_l = sigma_l
        self.min_gradient = min_gradient
        self.camera_model_key = camera_model_key
        self.image_key = image_key
        self.step_size_ref = step_size_ref
        self.image_grad = GradientImage(grad_x(image_key), grad_y(image_key))
        self.inv_depth_range = InvDepthSearchRange(*inv_depth_search_range)

    def __call__(self, camera_model_ref, image_ref, T_rk, u_key, prior):
        if prior.inv_depth <= 0:
            return prior, FLAG.NEGATIVE_PRIOR_DEPTH

        inv_depth_range = self.inv_depth_range(prior)
        if inv_depth_range is None:
            return prior, FLAG.HYPOTHESIS_OUT_OF_SERCH_RANGE

        x_key = self.camera_model_key.normalize(u_key)

        # step size / inv depth = approximately const
        inv_depth_ref = calc_ref_inv_depth(T_rk, x_key, prior.inv_depth)
        step_size_key = (prior.inv_depth / inv_depth_ref) * self.step_size_ref
        if inv_depth_ref <= 0:
            return prior, FLAG.NEGATIVE_REF_DEPTH

        xs_key = key_coordinates(get_translation(T_rk), x_key, step_size_key)
        us_key = self.camera_model_key.unnormalize(xs_key)
        if not all_points_in_image(us_key, self.image_key.shape):
            return prior, FLAG.KEY_OUT_OF_RANGE

        intensities_key = interpolation(self.image_key, us_key)
        gradient_key = calc_gradient_norm(intensities_key)

        if gradient_key < self.min_gradient:
            return prior, FLAG.INSUFFICIENT_GRADIENT

        x_range_ref = ref_search_range(T_rk, x_key,
                                       depth_search_range(*inv_depth_range))
        xs_ref = ref_coordinates(x_range_ref, self.step_size_ref)
        if len(xs_ref) < len(xs_key):
            return prior, FLAG.REF_EPIPOLAR_TOO_SHORT

        us_ref = camera_model_ref.unnormalize(xs_ref)

        if not is_in_image_range(us_ref[0], image_ref.shape):
            return prior, FLAG.REF_CLOSE_OUT_OF_RANGE

        # TODO when does this condition become true?
        if not is_in_image_range(us_ref[-1], image_ref.shape):
            return prior, FLAG.REF_FAR_OUT_OF_RANGE

        intensities_ref = interpolation(image_ref, us_ref)
        argmin = search_intensities(intensities_key, intensities_ref)
        depth_key = calc_key_depth(T_rk, x_key, xs_ref[argmin])

        x_max_ref, x_min_ref = x_range_ref
        direction = normalize_length(x_max_ref - x_min_ref)
        variance = calc_observation_variance(
            alpha=calc_alpha(T_rk, x_key, direction, prior.inv_depth),
            geo_variance=geometric_variance(
                key_epipolar_direction(get_translation(T_rk), x_key),
                self.image_grad(u_key), self.sigma_l, epsilon=1e-4),
            photo_variance=photometric_variance(gradient_key / step_size_key,
                                                self.sigma_i)
        )
        return Hypothesis(safe_invert(depth_key), variance), FLAG.SUCCESS


class AgeDependentValues(object):
    def __init__(self, age_map, values):
        self.age_map = age_map
        self.values = values

    def __call__(self, u):
        x, y = u
        age = self.age_map[y, x]
        if age == 0:
            return None

        return self.values[-age]


class InvDepthMapEstimator(object):
    def __init__(self, estimator):
        self._estimator = estimator

    def __call__(self, inv_depth_map, variance_map,
                 age_map, refframes):
        age_dict = AgeDependentValues(age_map, refframes)
        assert(inv_depth_map.shape == variance_map.shape == age_map.shape)

        flag_map = np.full(inv_depth_map.shape, FLAG.NOT_PROCESSED)
        for u_key in tqdm(image_coordinates(inv_depth_map.shape)):
            ref = age_dict(u_key)
            if ref is None:
                continue

            x, y = u_key
            prior = Hypothesis(inv_depth_map[y, x], variance_map[y, x])
            result, flag = self._estimator(*ref, u_key, prior)

            inv_depth_map[y, x] = result.inv_depth
            variance_map[y, x] = result.variance
            flag_map[y, x] = flag

        return inv_depth_map, variance_map, flag_map
