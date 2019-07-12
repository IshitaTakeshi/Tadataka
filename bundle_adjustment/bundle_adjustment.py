from autograd import numpy as np
from scipy.optimize import least_squares

from bundle_adjustment.triangulation import two_view_reconstruction
from bundle_adjustment.mask import keypoint_mask, point_mask, pose_mask
from optimization.robustifiers import SquaredRobustifier
from optimization.updaters import GaussNewtonUpdater
from optimization.optimizers import BaseOptimizer
from optimization.array_utils import Flatten, Reshape
from optimization.transformers import BaseTransformer
from optimization.errors import SumRobustifiedNormError
from optimization.functions import Function
from optimization.residuals import BaseResidual
from projection.projections import PerspectiveProjection
from rigid.rotation import rodrigues
from rigid.transformation import transform_each


class ParameterConverter(object):
    def __init__(self, n_viewpoints, n_points):
        self.n_viewpoints = n_viewpoints
        self.n_points = n_points

    def from_params(self, params):
        assert(params.shape[0] == self.n_viewpoints * 6 + self.n_points * 3)

        N = 3 * self.n_viewpoints
        omegas = params[0:N].reshape(self.n_viewpoints, 3)
        translations = params[N:2*N].reshape(self.n_viewpoints, 3)
        points = params[2*N:].reshape(self.n_points, 3)

        return omegas, translations, points

    def to_params(self, omegas, translations, points):
        return np.concatenate((
            omegas.flatten(),
            translations.flatten(),
            points.flatten()
        ))

    @property
    def n_dims(self):
        return 6 * self.n_viewpoints + 3 * self.n_points


class RigidTransform(Function):
    def compute(self, omegas, translations, points):
        return transform_each(rodrigues(omegas), translations, points)


class ScipyLeastSquaresOptimizer(BaseOptimizer):
    def __init__(self, updater, error):
        super().__init__(updater, error)

    def optimize(self, initial_theta):
        res = least_squares(self.updater.residual, initial_theta,
                            self.updater.jacobian,
                            loss=self.error.compute,
                            ftol=0.1, max_nfev=20, verbose=2)
        return res.x


def select_initial_viewpoints(keypoints):
    masks = keypoint_mask(keypoints)
    print(masks)
    n_visible = np.sum(masks, axis=1)
    print(n_visible)
    viewpoint1, viewpoint2 = np.argsort(n_visible)[::-1][0:2]
    mask = np.logical_and(masks[viewpoint1], masks[viewpoint2])
    return mask, viewpoint1, viewpoint2


def count_correspondences(mask_reference, masks):
    return np.sum(np.logical_and(mask_reference, masks), axis=1)


class BaseInitializer(object):
    def initialize(self):
        raise NotImplementedError()


class PointInitializer(BaseInitializer):
    def __init__(self, keypoints, K):
        self.keypoints = keypoints
        self.K = K

    def initialize(self):
        mask, viewpoint1, viewpoint2 = select_initial_viewpoints(
            self.keypoints
        )

        n_points = self.keypoints.shape[1]

        points = np.full((n_points, 3), np.nan)
        print(self.keypoints[viewpoint1, mask])
        print(self.keypoints[viewpoint2, mask])
        R, t, points_ = two_view_reconstruction(
            self.keypoints[viewpoint1, mask],
            self.keypoints[viewpoint2, mask],
            self.K
        )

        print(mask.shape)
        print(points.shape)
        points[mask] = points_
        return points


class PoseInitializer(BaseInitializer):
    def __init__(self, keypoints, K):
        self.keypoints = keypoints
        self.K = K

    def initialize(self, points):
        # at least 4 corresponding points have to be found
        # between keypoitns and 3D poitns
        required_correspondences = 4

        # TODO make independent from cv2
        import cv2
        n_viewpoints = self.keypoints.shape[0]

        omegas = np.empty((n_viewpoints, 3))
        translations = np.empty((n_viewpoints, 3))

        masks = np.logical_and(
            point_mask(points),
            keypoint_mask(self.keypoints)
        )
        for i in range(n_viewpoints):
            if np.sum(masks[i]) < required_correspondences:
                omegas[i] = np.nan
                translations[i] = np.nan
                continue

            retval, rvec, tvec = cv2.solvePnP(
                points[masks[i]], self.keypoints[i, masks[i]],
                self.K, np.zeros(4)
            )
            omegas[i] = rvec.flatten()
            translations[i] = tvec.flatten()
        return omegas, translations


class Initializer(object):
    def __init__(self, keypoints, K):
        self.point_initializer = PointInitializer(keypoints, K)
        self.pose_initializer = PoseInitializer(keypoints, K)

    def initialize(self):
        """
        Initialize 3D points and camera poses

        keypoints : np.ndarray
            A set of keypoints of shape (n_viewpoints, 2)
        """

        points = self.point_initializer.initialize()
        omegas, translations = self.pose_initializer.initialize(points)
        return omegas, translations, points


class Transformer(Function):
    def __init__(self, n_viewpoints, n_points, camera_parameters, converter):
        self.transform = RigidTransform()
        self.reshape1 = Reshape((n_viewpoints * n_points, 3))
        self.projection = PerspectiveProjection(camera_parameters)
        self.reshape2 = Reshape((n_viewpoints, n_points, 2))
        self.converter = converter

    def compute(self, params):
        omegas, translations, points = self.converter.from_params(params)
        points = self.transform.compute(omegas, translations, points)
        points = self.reshape1.compute(points)
        keypoints = self.projection.compute(points)
        keypoints = self.reshape2.compute(keypoints)
        return keypoints


class Error(Function):
    def __init__(self, residual, robustifier):
        self.residual = residual
        self.reshape = Reshape((-1, 2))
        self.error = SumRobustifiedNormError(robustifier)

    def compute(self, params):
        residual = self.residual.compute(params)
        residual = self.reshape.compute(residual)
        return self.error.compute(residual)


class Optimizer(object):
    def __init__(self, residual):
        robustifier = SquaredRobustifier()
        updater = GaussNewtonUpdater(residual, robustifier)
        error = Error(residual, robustifier)
        self.optimizer = BaseOptimizer(updater, error)

    def optimize(self, initial_params):
        return self.optimizer.optimize(initial_params)


class MaskedResidual(BaseResidual):
    def __init__(self, y, transformer, masks):
        super().__init__(y, transformer)
        self.masks = masks

    def compute(self, theta):
        residual = super().compute(theta)
        residual = residual[self.masks].flatten()
        return residual


def mask_params(omegas, translations, points):
    mask = pose_mask(omegas, translations)
    omegas, translations = omegas[mask], translations[mask]
    mask = point_mask(points)
    points = points[mask]
    return omegas, translations, points


class BundleAdjustment(object):
    def __init__(self, keypoints, camera_parameters):
        """
        keypoints: np.ndarray
            Keypoint coordinates of shape (n_viewpoints, n_points, 2)
        """

        self.initializer = Initializer(keypoints, camera_parameters.matrix)
        n_viewpoints, n_points = keypoints.shape[0:2]
        self.converter = ParameterConverter(n_viewpoints, n_points)

        transformer = Transformer(n_viewpoints, n_points, camera_parameters,
                                  self.converter)
        residual = MaskedResidual(keypoints, transformer,
                                  keypoint_mask(keypoints))
        self.optimizer = Optimizer(residual)

    def optimize(self):
        params = self.initializer.initialize()
        params = mask_params(*params)
        params = self.converter.to_params(*params)
        params = self.optimizer.optimize(params)
        return self.converter.from_params(params)
