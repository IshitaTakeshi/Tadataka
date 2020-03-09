import numpy as np
import numba

from tadataka.pose import LocalPose
from tadataka.vector import normalize_length
from tadataka.coordinates import image_coordinates
from tadataka.utils import is_in_image_range
from tadataka.matrix import to_homogeneous
from tadataka.projection import pi
from tadataka.interpolation import interpolation
from tadataka.triangulation import depth_from_triangulation, DepthsFromTriangulation
from tadataka.rigid_transform import transform, Warp3D
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.vo.semi_dense.epipolar import (
    reference_coordinates, key_coordinates
)
from tadataka.vo.semi_dense.variance import (
    photometric_variance, geometric_variance, calc_alpha
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


def calc_depth_ref(warp_key_to_ref, x_key, depth_key):
    q = warp_key_to_ref(depth_key * to_homogeneous(x_key))
    return q[2]


def step_size_ratio(warp_key_to_ref, inv_depth_key, x_key):
    depth_key = invert_depth(inv_depth_key)
    depth_ref = calc_depth_ref(warp_key_to_ref, x_key, depth_key)
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


def epipolar_search_range_(warp_key_to_ref, x_key, depth_range):
    min_depth, max_depth = depth_range
    x = to_homogeneous(x_key)
    x_ref_min = pi(warp_key_to_ref(x * min_depth))
    x_ref_max = pi(warp_key_to_ref(x * max_depth))
    return x_ref_min, x_ref_max


def epipolar_search_range(warp_key_to_ref, x_key, inv_depth_range):
    depth_range = depth_search_range(inv_depth_range)
    return epipolar_search_range_(warp_key_to_ref, x_key, depth_range)


def unnormalize_in_image(camera_model, xs, coordinate_range):
    us = camera_model.unnormalize(xs)
    mask = is_in_image_range(us, coordinate_range)
    return xs[mask], us[mask]


class Uncertaintity(object):
    def __init__(self, sigma_i, sigma_l):
        self.sigma_l = sigma_l
        self.sigma_i = sigma_i

    def __call__(self, x_key, x_ref, x_range_ref, R, t,
                 image_grad, epipolar_gradient):
        x_min_ref, x_max_ref = x_range_ref
        direction = normalize_length(x_max_ref - x_min_ref)
        alpha = calc_alpha(x_key, x_ref, direction, R, t)
        geo_var = geometric_variance(x_key, pi(t), image_grad, self.sigma_l)
        photo_var = photometric_variance(epipolar_gradient, self.sigma_i)
        return calc_observation_variance(alpha, geo_var, photo_var)


class InverseDepthEstimator(object):
    def __init__(self, keyframe, sigma_i, sigma_l, step_size_ref,
                 min_gradient):
        assert(np.ndim(keyframe.image) == 2)
        self.pose_key = keyframe.pose
        self.image_key = keyframe.image
        self.camera_model_key = keyframe.camera_model
        self.min_gradient = min_gradient

        # -1 for bilinear interpolation
        height, width = self.image_key.shape
        self.coordinate_range = (height-1, width-1)

        self.uncertaintity = Uncertaintity(sigma_i, sigma_l)
        self.step_size_ref = step_size_ref
        self.image_grad = GradientImage(
            grad_x(self.image_key), grad_y(self.image_key)
        )
        self.search_range = InverseDepthSearchRange(
            min_inv_depth=0.1, max_inv_depth=5.0
        )

    def __call__(self, refframe, u_key, prior_inv_depth, prior_variance):
        pose_ref = refframe.pose
        camera_model_ref = refframe.camera_model
        image_ref = refframe.image

        from matplotlib import pyplot as plt

        # assert(isinstance(pose_key_to_ref, LocalPose))
        warp = Warp3D(self.pose_key, pose_ref)

        x_key = self.camera_model_key.normalize(u_key)
        x_ref = pi(warp(invert_depth(prior_inv_depth) * to_homogeneous(x_key)))
        u_ref = camera_model_ref.unnormalize(x_ref)

        if False:
            fig = plt.figure()

            ax = fig.add_subplot(121)
            ax.set_title("keyframe")
            ax.imshow(self.image_key, cmap="gray")
            ax.scatter(u_key[0], u_key[1], c="red")

            ax = fig.add_subplot(122)
            ax.set_title("reference frame")
            ax.imshow(image_ref, cmap="gray")
            ax.scatter(u_ref[0], u_ref[1], c="red")
            plt.show()

        ratio = step_size_ratio(warp, prior_inv_depth, x_key)
        step_size_key = ratio * self.step_size_ref

        # image of reference camera center on the keyframe
        x_ref = pi(np.dot(self.pose_key.R.T, pose_ref.t - self.pose_key.t))
        xs_key = key_coordinates(x_key, x_ref, step_size_key)
        us_key = self.camera_model_key.unnormalize(xs_key)

        if not is_in_image_range(us_key, self.coordinate_range).all():
            return prior_inv_depth # , prior_variance

        inv_depth_range = self.search_range(prior_inv_depth, prior_variance)
        x_range_ref = epipolar_search_range(warp, x_key, inv_depth_range)

        xs_ref = reference_coordinates(x_range_ref, self.step_size_ref)
        xs_ref, us_ref = unnormalize_in_image(camera_model_ref, xs_ref,
                                              self.coordinate_range)

        if len(xs_ref) < len(xs_key):
            return prior_inv_depth # , prior_variance

        intensities_key = interpolation(self.image_key, us_key)
        epipolar_gradient = intensity_gradient(intensities_key, step_size_key)
        if epipolar_gradient < self.min_gradient:
            return prior_inv_depth # , prior_variance

        intensities_ref = interpolation(image_ref, us_ref)
        argmin = search_intensities(intensities_key, intensities_ref)

        R0, t0 = self.pose_key.R, self.pose_key.t
        R1, t1 = pose_ref.R, pose_ref.t
        R = np.dot(R1.T, R0)
        t = np.dot(R1.T, t0 - t1)
        depth_key = depth_from_triangulation(R, t, x_key, xs_ref[argmin])

        return invert_depth(depth_key)

        # R = np.dot(pose_ref.R.T, self.pose_key.R)
        # t = np.dot(pose_ref.R.T, self.pose_key.t - pose_ref.t)
        # variance = self.uncertaintity(x_key, x_ref, x_range_ref, R, t,
        #                               image_grad, epipolar_gradient)
        # return invert_depth(depth_key), variance
