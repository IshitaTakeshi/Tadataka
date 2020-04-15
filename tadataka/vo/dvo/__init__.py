import sys
import warnings

import numpy as np
from numpy.linalg import norm

from skimage.transform import rescale
from skimage.color import rgb2gray

from tadataka import camera
from tadataka.metric import PhotometricError
from tadataka.coordinates import image_coordinates
from tadataka.math import solve_linear_equation
from tadataka.utils import is_in_image_range
from tadataka.projection import inv_pi, pi
from tadataka.camera import CameraModel, CameraParameters
from tadataka.rigid_transform import transform
from tadataka.interpolation import interpolation
from tadataka.vo.dvo.jacobian import calc_image_gradient, calc_jacobian
from tadataka.pose import WorldPose
from tadataka.robust.weights import (compute_weights_huber,
                                     compute_weights_student_t,
                                     compute_weights_tukey)


def calc_error(r, weights=None):
    if weights is None:
        return np.dot(r, r)
    return np.dot(r * weights, r)  # r * W * r


def compute_weights(name, residuals):
    if name == "tukey":
        return compute_weights_tukey(residuals)
    if name == "student-t":
        return compute_weights_student_t(residuals)
    if name == "huber":
        return compute_weights_huber(residuals)
    raise ValueError(f"No such weights '{name}'")


def level_to_scale(level):
    return 1 / pow(1.2, level)


def calc_pose_update(camera_model1, residuals, GX1, GY1, P1, weights):
    assert(GX1.shape == GY1.shape)
    us1 = camera_model1.unnormalize(pi(P1))
    mask = is_in_image_range(us1, GX1.shape) & (P1[:, 2] > 0)

    if not np.any(mask):
        # warped coordinates are out of image range
        return None

    r = residuals[mask]
    p1 = P1[mask]
    gx1 = interpolation(GX1, us1[mask])
    gy1 = interpolation(GY1, us1[mask])

    J = calc_jacobian(camera_model1.camera_parameters.focal_length,
                      gx1, gy1, p1)

    if weights is None:
        return solve_linear_equation(J, r)
    if isinstance(weights, str):
        weights = compute_weights(weights, r)
        return solve_linear_equation(J, r, weights)

    weights = weights.flatten()[mask]
    return solve_linear_equation(J, r, weights)


class _PoseChangeEstimator(object):
    def __init__(self, camera_model0, camera_model1, max_iter):
        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1
        self.max_iter = max_iter

    def __call__(self, I0, D0, I1, pose10, weights=None):
        def warn():
            warnings.warn("Camera pose change is too large.", RuntimeWarning)

        error = PhotometricError(self.camera_model0, self.camera_model1,
                                 I0, D0, I1)

        us0 = image_coordinates(I0.shape)
        xs0 = self.camera_model0.normalize(us0)
        P0 = inv_pi(xs0, D0.flatten())
        GX1, GY1 = calc_image_gradient(I1)
        residuals = (I0 - I1).flatten()

        prev_error = error(pose10)
        for k in range(self.max_iter):
            P1 = transform(pose10.R, pose10.t, P0)
            xi = calc_pose_update(self.camera_model1, residuals,
                                  GX1, GY1, P1, weights)
            if xi is None:
                warn()
                return pose10

            dpose = WorldPose.from_se3(xi)
            candidate = dpose * pose10

            curr_error = error(candidate)
            if curr_error > prev_error:
                break
            prev_error = curr_error

            print("norm(xi) = {:.6f}  error = {:.6f}".format(norm(xi), curr_error))

            pose10 = candidate
        return pose10


class PoseChangeEstimator(object):
    def __init__(self, camera_model0, camera_model1,
                 n_coarse_to_fine=5):
        self.n_coarse_to_fine = n_coarse_to_fine

        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1

    def __call__(self, I0, D0, I1, weights=None, pose10=WorldPose.identity()):
        assert(I0.shape == D0.shape == I1.shape)
        assert(np.ndim(I0) == 2)
        assert(np.ndim(D0) == 2)
        assert(np.ndim(I1) == 2)

        print("\n")
        for level in list(reversed(range(self.n_coarse_to_fine))):
            pose10 = self._estimate_at(pose10, level, I0, D0, I1, weights)

        return pose10

    def _estimate_at(self, prior, level, I0, D0, I1, W0):
        print("level =", level)
        scale = level_to_scale(level)

        camera_model0 = camera.resize(self.camera_model0, scale)
        camera_model1 = camera.resize(self.camera_model1, scale)
        estimator = _PoseChangeEstimator(camera_model0, camera_model1,
                                         max_iter=20)

        D0 = rescale(D0, scale)
        I0 = rescale(I0, scale)
        I1 = rescale(I1, scale)
        if isinstance(W0, np.ndarray):
            W0 = rescale(W0, scale)

        return estimator(I0, D0, I1, prior, W0)


class DVO(object):
    def __init__(self):
        self.pose = WorldPose.identity()
        self.frames = []

    def estimate(self, frame1):
        if len(self.frames) == 0:
            self.frames.append(frame1)
            return self.pose

        frame0 = self.frames[-1]

        I0, D0 = rgb2gray(frame0.image), frame0.depth_map
        I1 = rgb2gray(frame1.image)

        estimator = PoseChangeEstimator(frame1.camera_model, I0, D0, I1)
        dpose = estimator.estimate()

        self.frames.append(frame1)

        self.pose = dpose * self.pose
        return self.pose
