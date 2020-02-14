import sys
import warnings

import numpy as np
from numpy.linalg import norm

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

from tadataka.camera import CameraParameters
from tadataka.rigid_transform import transform
from tadataka.interpolation import interpolation
from tadataka.se3 import exp_se3, get_rotation, get_translation
from tadataka.vo.dvo.mask import compute_mask
from tadataka.vo.dvo.projection import inverse_projection, projection
from tadataka.vo.dvo.jacobian import calc_image_gradient, calc_jacobian
from tadataka.pose import Pose
from tadataka.robust.weights import (compute_weights_huber,
                                     compute_weights_student_t,
                                     compute_weights_tukey)


def calc_error(r, weights=None):
    if weights is None:
        return np.dot(r, r)
    return np.dot(r * weights, r)  # r * W * r


def solve_linear_equation(J, r, weights=None):
    if weights is not None:
        # mulitply weights to each row
        J = J * weights.reshape(-1, 1)

    xi, errors, _, _ = np.linalg.lstsq(J, r, rcond=None)
    error = 0 if len(errors) == 0 else errors[0]
    return xi, error


def level_to_ratio(level):
    return 1 / pow(2, level)


def calc_pose_update(camera_parameters,
                     I0, D0, I1, DX, DY, S, pose, min_depth=1e-8):

    # Transform onto the t0 coordinate
    # means that
    # 1. backproject each pixel in the t0 frame to 3D
    # 2. transform the 3D points to t1 coordinates
    # 3. reproject the transformed 3D points to the t1 coordinates
    # 4. interpolate image gradient maps using the reprojected coordinates

    P = transform(pose.R, pose.t, S)  # to t1 coordinates
    Q = projection(camera_parameters, P)
    mask = compute_mask(D0, Q).flatten()

    if not np.any(mask):
        # return xi = np.zeros(6), error = np.nan
        # if there is no valid pixel
        return np.zeros(6), np.nan

    P = P[mask]
    I0 = I0.flatten()[mask]  # you don't need to warp I0
    I1 = interpolation(I1, Q, order=1)[mask]
    DX = interpolation(DX, Q, order=1)[mask]
    DY = interpolation(DY, Q, order=1)[mask]

    # J.shape == (n_image_pixels, 6)
    J = calc_jacobian(camera_parameters, DX, DY, P)

    r = -(I1 - I0)
    # weights = compute_weights_tukey(r)
    weights = compute_weights_student_t(r)
    print(J.shape, r.shape, weights.shape)

    xi, error = solve_linear_equation(J, r, weights)
    return xi, error


class PoseChangeEstimator(object):
    def __init__(self, camera_model, I0, D0, I1,
                 epsilon=1e-4, max_iter=20):
        # TODO check if np.ndim(D0) == np.ndim(I1) == 2

        self.I0 = I0
        self.D0 = D0
        self.I1 = I1

        self.epsilon = epsilon
        self.max_iter = max_iter

        # FIXME use 'camera_model'
        self.camera_parameters = camera_model.camera_parameters

    def estimate(self, n_coarse_to_fine=5):
        levels = list(reversed(range(n_coarse_to_fine)))

        pose = Pose.identity()
        for level in levels:
            try:
                pose = self.estimate_motion_at(level, pose)
            except np.linalg.LinAlgError as e:
                sys.stderr.write(str(e) + "\n")
                return Pose.identity()
        # invert because 'pose' is representing the pose change from t1 to t0
        return pose.inv()

    def camera_parameters_at(self, level):
        """Change camera parameters as the image is resized"""
        focal_length = self.camera_parameters.focal_length
        offset = self.camera_parameters.offset
        ratio = level_to_ratio(level)
        return CameraParameters(focal_length * ratio, offset * ratio)

    def image_shape_at(self, level):
        shape = np.array(self.I0.shape)
        ratio = level_to_ratio(level)
        return shape * ratio

    def estimate_motion_at(self, level, pose):
        camera_parameters = self.camera_parameters_at(level)
        shape = self.image_shape_at(level)

        I0 = resize(self.I0, shape)
        D0 = resize(self.D0, shape)
        I1 = resize(self.I1, shape)

        S = inverse_projection(camera_parameters, D0)

        DX, DY = calc_image_gradient(I1)

        for k in range(self.max_iter):
            dxi, error = calc_pose_update(
                camera_parameters,
                I0, D0, I1, DX, DY, S, pose
            )

            if np.isnan(error):
                warnings.warn(
                    "There's no valid pixel at level {}. "\
                    "Camera's pose change is too large ".format(level),
                    RuntimeWarning
                )

            if norm(dxi) < self.epsilon:
                break

            dpose = Pose.from_se3(dxi)
            pose = pose * dpose
        return pose


class DVO(object):
    def __init__(self):
        self.pose = Pose.identity()
        self.frames = []

    def estimate(self, frame1):
        if len(self.frames) == 0:
            self.frames.append(frame1)
            return self.pose

        frame0 = self.frames[-1]

        I0, D0 = rgb2gray(frame0.image), frame0.depth_map,
        I1 = rgb2gray(frame1.image)

        estimator = PoseChangeEstimator(frame1.camera_model, I0, D0, I1)
        dpose = estimator.estimate()

        self.frames.append(frame1)

        self.pose = self.pose * dpose
        return self.pose
