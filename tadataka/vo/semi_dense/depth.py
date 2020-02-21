import numpy as np
from tadataka.rigid_transform import Transform
from tadataka.triangulation import DepthFromTriangulation
from tadataka.matrix import to_homogeneous
from tadataka.projection import pi
from tadataka.interpolation import interpolation
from tadataka.vo.semi_dense.epipolar import (
    ReferenceCoordinates, EpipolarDirection
)

def invert_depth(depth, EPSILON=1e-16):
    return 1 / (depth + EPSILON)


def depth_search_range(inv_depth_range):
    min_inv_depth, max_inv_depth = inv_depth_range
    min_depth = invert_depth(max_inv_depth)
    max_depth = invert_depth(min_inv_depth)
    return min_depth, max_depth


def epipolar_search_range(transform, x_key, depth_range):
    min_depth, max_depth = depth_range
    x_ref_min = pi(transform(to_homogeneous(x_key) * min_depth))
    x_ref_max = pi(transform(to_homogeneous(x_key) * max_depth))
    return x_ref_min, x_ref_max


def calc_depth_ref(transform, x_key, depth_key):
    q = transform(depth_key * to_homogeneous(x_key))
    return q[2]


def step_size_ratio(transform, x_key, inv_depth_key):
    depth_key = invert_depth(inv_depth_key)
    depth_ref = calc_depth_ref(transform, x_key, depth_key)
    inv_depth_ref = invert_depth(depth_ref)
    return inv_depth_key / inv_depth_ref


def estimate_inv_depth(estimator, transform, inv_depth_range, x_key):
    depth_range = depth_search_range(inv_depth_range)
    x_range = epipolar_search_range(transform, x_key, depth_range)
    depth = estimator(x_key, x_range)
    return invert_depth(depth)


def inv_depth_search_range(inv_depth, variance, deviation_factor=2.0):
    return (inv_depth - deviation_factor * variance,
            inv_depth + deviation_factor * variance)


class KeyStepSize(object):
    def __init__(self, transform, step_size_ref):
        self.transform = transform
        self.step_size_ref = step_size_ref

    def __call__(self, x_key, inv_depth):
        ratio = step_size_ratio(self.transform, x_key, inv_depth)
        return ratio * self.step_size_ref


class GradientImage(object):
    def __init__(self, image_grad_x, image_grad_y):
        self.grad_x = image_grad_x
        self.grad_y = image_grad_y

    def __call__(self, u_key):
        gx = interpolation(self.grad_x, u_key, order=1)
        gy = interpolation(self.grad_y, u_key, order=1)
        return np.array([gx, gy])


class VarianceEstimator(object):
    def __init__(self, epipolar_direction, sigma_l, sigma_i):
        self.sigma_l = sigma_l
        self.sigma_i = sigma_i

    def __call__(self, x, gradient, epipolar_gradient, alpha):
        direction = self.epipolar_direction(x)
        sigma_g = geometric_variance(direction, gradient, self.sigma_l)
        sigma_p = photometric_variance(epipolar_gradient, self.sigma_i)
        return alpha * alpha * (sigma_g + sigma_p)


def calc_observation_variance(alpha, geo_variance, photo_variance):
    return alpha * alpha * (geo_variance + photo_variance)


class InverseDepthMapEstimator(object):
    def __init__(self, step_size_ref):
        epipolar_direction = EpipolarDirection(t)
        self.image_grad = GradientImage(grad_x(image_key), grad_y(image_key))
        self.variance = VarianceEstimator(epipolar_direction)
        self.key_coordinates = KeyCoordinates(camera_model_key,
                                              epipolar_direction)
        self.photo_variance = PhotometricVariance(sigma_i)
        self.geo_variance = GeometricVariance(epipolar_direction, sigma_l)
        self.transform = Transform(pose_key_to_ref.R, pose_key_to_ref.t)
        self.ref_coordinates = ReferenceCoordinates(camera_model_ref,
                                                    image_ref, step_size_ref)
        self.calc_depths = DepthFromTriangulation(Pose.identity(),
                                                  pose_key_to_ref)
        self.step_size_key = KeyStepSize(self.transform, step_size_ref)

    def _estimate(self, x_key, u_key, prior_inv_depth):
        inv_depth_range = inv_depth_search_range(inv_depth1, variance1)
        depth_range = depth_search_range(inv_depth_range)
        x_range_ref = epipolar_search_range(self.transform, x_key, depth_range)

        xs_ref, us_ref = self.ref_coordinates(x_range_ref)
        intensities_key = interpolation(self.image_key, us_key)
        intensities_ref = interpolation(self.image_ref, us_ref)
        index = search_intensities(intensities_key, intensities_ref)
        x_ref = xs_ref[index]

        key_depth, ref_depth = self.calc_depths(x_key, x_ref)

        step_size_key = self.step_size_key(x_key, prior_inv_depth)

        us_key = self.key_coordinates(x, step_size_key)
        mask = is_in_image_range(us, self.image_shape)

        alpha = calc_alpha(x_key, x_ref, R, t,
                           search_step_ref * epipolar_direction)

        epipolar_gradient = intensity_gradient(intensities_key, step_size_key)
        geo_var = self.geo_variance(x_key, self.image_grad(u_key))
        photo_var = self.photo_variance(epipolar_gradient)
        variance = calc_observation_variance(alpha, geo_var, photo_var)
        return invert_depth(key_depth), variance

    def __call__(self, inv_depth_map, variance_map):
        us_key = image_coordinates(image_shape)
        xs_key = self.camera_model_key.normalize(us_key)

        for x_key, u_key in zip(xs_key, us_key):
            u, v = u_key
            inv_depth1 = inv_depth_map[v, u]
            variance1 = variance_map[v, u]

            inv_depth2, variance2 = self._estimate(x_key, u_key, inv_depth1)
            inv_depth, variance = update(inv_depth1, inv_depth2,
                                         variance1, variance2)
            inv_depth_map[v, u] = inv_depth
            variance_map[v, u] = variance
        return inv_depth_map, variance_map
