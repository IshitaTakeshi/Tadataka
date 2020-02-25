import numpy as np
import numba

from tadataka.utils import is_in_image_range
from tadataka.matrix import to_homogeneous
from tadataka.projection import pi
from tadataka.interpolation import interpolation
from tadataka.triangulation import DepthFromTriangulation
from tadataka.pose import Pose
from tadataka.rigid_transform import Transform
from tadataka.vo.semi_dense.epipolar import (
    ReferenceCoordinates, EpipolarDirection, KeyCoordinates
)
from tadataka.vo.semi_dense.variance import (
    PhotometricVariance, GeometricVariance, Alpha
)
from tadataka.vo.semi_dense.intensities import search_intensities
from tadataka.gradient import grad_x, grad_y


def intensity_gradient(intensities, interval):
    return np.linalg.norm(intensities[1:] - intensities[:-1]) / interval


def calc_inv_depths(ref_coordinate, key_coordinate,
                    R_key_to_ref, t_key_to_ref):
    rot_inv_x = np.dot(R_key_to_ref, x_key)

    n = x_ref[0:2] * t_key_to_ref - t_key_to_ref[0:2]
    d = rot_inv_x[0:2] - ref_coordinate * rot_inv_x[2]
    return d / n


def depth_coordinate(search_step):
    return 0 if np.abs(search_step[0]) > np.abs(search_step[1]) else 1


class InsufficientCoordinatesError(Exception):
    pass


def error_if_insufficient_coordinates(mask, min_coordinates):
    n_in_image = np.sum(mask)
    if n_in_image < min_coordinates:
        raise InsufficientCoordinatesError(
            "Insufficient number of coordinates "
            "sampled from the epipolar line. "
            "Required {}, but found {}.".format(min_coordinates, n_in_image)
        )


def update(d1, d2, var1, var2):
    d = (d1 * var2 + d2 * var1) / (var1 + var2)
    var = (var1 * var2) / (var1 + var2)
    return d, var


def calc_observation_variance(alpha, geo_variance, photo_variance):
    return alpha * alpha * (geo_variance + photo_variance)


def invert_depth(depth, EPSILON=1e-16):
    return 1 / (depth + EPSILON)


class InverseDepthSearchRange(object):
    def __init__(self, factor=2.0, min_inv_depth=1e-16):
        assert(min_inv_depth > 0.0)
        self.factor = factor
        self.min_inv_depth = min_inv_depth

    def __call__(self, inv_depth, variance):
        assert(variance >= 0.0)
        L = max(inv_depth - self.factor * variance, self.min_inv_depth)
        U = inv_depth + self.factor * variance
        return L, U


def calc_depth_ref(transform, x_key, depth_key):
    q = transform(depth_key * to_homogeneous(x_key))
    return q[2]


def step_size_ratio(transform, x_key, inv_depth_key):
    depth_key = invert_depth(inv_depth_key)
    depth_ref = calc_depth_ref(transform, x_key, depth_key)
    inv_depth_ref = invert_depth(depth_ref)
    return inv_depth_key / inv_depth_ref


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
        gx = interpolation(self.grad_x, u_key)
        gy = interpolation(self.grad_y, u_key)
        return np.array([gx, gy])


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


class InverseDepthEstimator(object):
    def __init__(self, pose_key_to_ref, image_key, image_ref,
                 camera_model_key, camera_model_ref,
                 sigma_i, sigma_l, step_size_ref):
        self.image_key, self.image_ref = image_key, image_ref
        R, t = pose_key_to_ref.R, pose_key_to_ref.t
        self.transform = Transform(R, t)
        epipolar_direction = EpipolarDirection(t)
        self.alpha = Alpha(R, t)
        self.image_grad = GradientImage(grad_x(image_key), grad_y(image_key))
        self.key_coordinates = KeyCoordinates(camera_model_key,
                                              epipolar_direction)
        self.photo_variance = PhotometricVariance(sigma_i)
        self.geo_variance = GeometricVariance(epipolar_direction, sigma_l)
        self.ref_coordinates = ReferenceCoordinates(camera_model_ref,
                                                    image_ref.shape,
                                                    step_size_ref)
        self.calc_depth = DepthFromTriangulation(pose_key_to_ref)
        self.step_size_key = KeyStepSize(self.transform, step_size_ref)

    def __call__(self, x_key, u_key, prior_inv_depth, prior_variance):
        inv_depth_range = inv_depth_search_range(prior_inv_depth,
                                                 prior_variance)
        depth_range = depth_search_range(inv_depth_range)
        x_range_ref = epipolar_search_range(self.transform, x_key, depth_range)

        xs_ref, us_ref = self.ref_coordinates(x_range_ref)
        step_size_key = self.step_size_key(x_key, prior_inv_depth)
        us_key = self.key_coordinates(x_key, step_size_key)

        intensities_key = interpolation(self.image_key, us_key)
        intensities_ref = interpolation(self.image_ref, us_ref)
        argmin = search_intensities(intensities_key, intensities_ref)
        x_ref = xs_ref[argmin]

        key_depth = self.calc_depth(x_key, x_ref)

        alpha = self.alpha(x_key, x_ref, x_range_ref)

        epipolar_gradient = intensity_gradient(intensities_key, step_size_key)
        geo_var = self.geo_variance(x_key, self.image_grad(u_key))
        photo_var = self.photo_variance(epipolar_gradient)
        variance = calc_observation_variance(alpha, geo_var, photo_var)
        return invert_depth(key_depth), variance


def estimate(self, inv_depth_map, variance_map):
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
