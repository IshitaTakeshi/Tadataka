from autograd import numpy as np
from scipy.optimize import least_squares

from bundle_adjustment.triangulation import two_view_reconstruction
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


def count_shared(mask1, mask2):
    return np.sum(np.logical_and(mask1, mask2))


def select_initial_viewpoints(masks):
    n_visible = np.sum(masks, axis=1)
    viewpoint1, viewpoint2 = np.argsort(n_visible)[::-1][0:2]
    mask = np.logical_and(
        masks[viewpoint1],
        masks[viewpoint2]
    )
    return mask, viewpoint1, viewpoint2


class Initializer(object):
    def __init__(self, keypoints, masks, K):
        n_viewpoints, n_points = keypoints.shape[0:2]
        self.keypoints = keypoints
        self.masks = masks
        self.K = K

    def initialize(self):
        """
        Initialize 3D points and camera poses

        keypoints : np.ndarray
            A set of keypoints of shape (n_viewpoints, 2)
        """

        n_viewpoints = self.keypoints.shape[0]
        reconstruction_mask, viewpoint1, viewpoint2 =\
            select_initial_viewpoints(self.masks)

        R, t, points = two_view_reconstruction(
            self.keypoints[viewpoint1, reconstruction_mask],
            self.keypoints[viewpoint2, reconstruction_mask],
            self.K
        )

        # TODO make independent from cv2
        import cv2

        omegas = np.empty((n_viewpoints, 3))
        translations = np.empty((n_viewpoints, 3))
        for i in range(n_viewpoints):
            # at least 3 points have to be seen from each viewpoint
            # to execute pnp properly
            n_visible = count_shared(reconstruction_mask, self.masks[i])
            assert(n_visible > 3)

            retval, rvec, tvec = cv2.solvePnP(points, self.keypoints[i],
                                              self.K, np.zeros(4))
            omegas[i] = rvec.flatten()
            translations[i] = tvec.flatten()
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
        print(residual.shape)
        print(self.masks.shape)
        residual = residual[self.masks].flatten()
        print(residual.shape)
        return residual


class BundleAdjustment(object):
    def __init__(self, keypoints, masks, camera_parameters):
        """
        keypoints: np.ndarray
            Keypoint coordinates of shape (n_viewpoints, n_points, 2)
        """

        self.initializer = Initializer(keypoints, masks, camera_parameters.matrix)
        n_viewpoints, n_points = keypoints.shape[0:2]
        self.converter = ParameterConverter(n_viewpoints, n_points)

        transformer = Transformer(n_viewpoints, n_points, camera_parameters,
                                  self.converter)
        residual = MaskedResidual(keypoints, transformer, masks)
        self.optimizer = Optimizer(residual)

    def optimize(self):
        params = self.converter.to_params(*self.initializer.initialize())
        params = self.optimizer.optimize(params)
        return self.converter.from_params(params)
