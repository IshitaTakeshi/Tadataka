import sys
import warnings

import numpy as np
from numpy.linalg import norm

from skimage.transform import resize
from skimage.color import rgb2gray

from tadataka.warp import Warp2D
from tadataka.metric import photometric_error
from tadataka.coordinates import image_coordinates
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


def solve_linear_equation(J, r, weights=None):
    if weights is not None:
        # mulitply weights to each row
        J = J * weights.reshape(-1, 1)

    xi, errors, _, _ = np.linalg.lstsq(J, r, rcond=None)
    return xi


def level_to_ratio(level):
    return 1 / pow(2, level)


def calc_pose_update(camera_model1, residuals, GX1, GY1, P1, weight_map):
    assert(GX1.shape == GY1.shape)
    us1 = camera_model1.unnormalize(pi(P1))
    mask = is_in_image_range(us1, GX1.shape)

    if not np.any(mask):
        # warped coordinates are out of image range
        return None

    r = residuals[mask]
    p1 = P1[mask]
    gx1 = interpolation(GX1, us1[mask])
    gy1 = interpolation(GY1, us1[mask])

    if weight_map is None:
        # weights = compute_weights_tukey(r)
        weights = compute_weights_student_t(r)
    else:
        weights = weight_map.flatten()[mask]

    J = calc_jacobian(camera_model1.camera_parameters.focal_length,
                      gx1, gy1, p1)
    xi = solve_linear_equation(J, r, weights)
    return xi


def image_shape_at(level, shape):
    ratio = level_to_ratio(level)
    return (int(shape[0] * ratio), int(shape[1] * ratio))


def camera_model_at(level, camera_model):
    """Change camera parameters as the image is resized"""
    ratio = level_to_ratio(level)
    params = camera_model.camera_parameters
    return CameraModel(
        CameraParameters(params.focal_length * ratio, params.offset * ratio),
        camera_model.distortion_model
    )


class _PoseChangeEstimator(object):
    def __init__(self, camera_model0, camera_model1, max_iter):
        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1
        self.max_iter = max_iter

    def _error(self, I0, D0, I1, pose10):
        return photometric_error(
            Warp2D(self.camera_model0, self.camera_model1,
                   pose10, WorldPose.identity()),
            I0, D0, I1
        )

    def __call__(self, I0, D0, I1, pose10, weight_map=None):
        def warn():
            warnings.warn("Camera pose change is too large.", RuntimeWarning)

        us0 = image_coordinates(I0.shape)
        xs0 = self.camera_model0.normalize(us0)
        P0 = inv_pi(xs0, D0.flatten())
        GX1, GY1 = calc_image_gradient(I1)
        residuals = (I0 - I1).flatten()

        prev_error = self._error(I0, D0, I1, pose10)
        for k in range(self.max_iter):
            P1 = transform(pose10.R, pose10.t, P0)
            xi = calc_pose_update(self.camera_model1, residuals,
                                  GX1, GY1, P1, weight_map)

            if xi is None:
                warn()
                return pose10

            dpose = WorldPose.from_se3(xi)
            canditate = dpose * pose10

            E = self._error(I0, D0, I1, canditate)
            if E > prev_error:
                break
            prev_error = E

            pose10 = canditate
        return pose10


class PoseChangeEstimator(object):
    def __init__(self, camera_model0, camera_model1,
                 n_coarse_to_fine=5, max_iter=10):
        self.n_coarse_to_fine = n_coarse_to_fine
        self.max_iter = max_iter

        self.camera_model0 = camera_model0
        self.camera_model1 = camera_model1

    def __call__(self, I0, D0, I1, weight_map=None,
                 pose10=WorldPose.identity()):
        W0 = weight_map
        assert(I0.shape == D0.shape == I1.shape)
        assert(np.ndim(I0) == 2)
        assert(np.ndim(D0) == 2)
        assert(np.ndim(I1) == 2)

        # transforms point in the 0th camera coordinate to
        # the 1st camera coordicoordinate

        for level in list(reversed(range(self.n_coarse_to_fine))):
            try:
                print("before ", photometric_error(
                    Warp2D(self.camera_model0, self.camera_model1,
                           pose10, WorldPose.identity()),
                    I0, D0, I1)
                )

                pose10 = self._estimate_at(pose10, level, I0, D0, I1, W0)

                print("after  ", photometric_error(
                    Warp2D(self.camera_model0, self.camera_model1,
                           pose10, WorldPose.identity()),
                    I0, D0, I1)
                )

            except np.linalg.LinAlgError as e:
                sys.stderr.write(str(e) + "\n")
                return WorldPose.identity()
        return pose10

    def _estimate_at(self, prior, level, I0, D0, I1, W0):
        camera_model0 = camera_model_at(level, self.camera_model0)
        camera_model1 = camera_model_at(level, self.camera_model1)
        estimator = _PoseChangeEstimator(camera_model0, camera_model1,
                                         max_iter=10)

        shape = image_shape_at(level, D0.shape)
        D0 = resize(D0, shape)
        I0 = resize(I0, shape)
        I1 = resize(I1, shape)

        weight_map = np.ones(shape)
        return estimator(I0, D0, I1, prior, weight_map)


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
