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


def initialize_poses(points, keypoints, K):
    # TODO make independent from cv2
    import cv2

    n_viewpoints = keypoints.shape[0]

    omegas = np.empty((n_viewpoints, 3))
    translations = np.empty((n_viewpoints, 3))
    for i in range(n_viewpoints):
        retval, rvec, tvec = cv2.solvePnP(points, keypoints[i], K, np.zeros(4))
        omegas[i] = rvec.flatten()
        translations[i] = tvec.flatten()
    return omegas, translations


def select_initial_viewpointns():
    return 0, 1


class Initializer(object):
    def __init__(self, keypoints, K):
        n_viewpoints, n_points = keypoints.shape[0:2]
        self.keypoints = keypoints
        self.K = K

    def initialize(self):
        """
        Initialize 3D points and camera poses

        keypoints : np.ndarray
            A set of keypoints of shape (n_viewpoints, 2)
        """

        viewpoint1, viewpoint2 = select_initial_viewpointns()

        R, t, points = two_view_reconstruction(
            self.keypoints[viewpoint1],
            self.keypoints[viewpoint2],
            self.K
        )

        omegas, translations = initialize_poses(
            points, self.keypoints, self.K)

        return omegas, translations, points


class Transformer(Function):
    def __init__(self, n_viewpoints, n_points, camera_parameters, converter):
        self.transform = RigidTransform()
        self.reshape = Reshape((n_viewpoints * n_points, 3))
        self.projection = PerspectiveProjection(camera_parameters)
        self.flatten = Flatten()
        self.converter = converter

    def compute(self, params):
        omegas, translations, points = self.converter.from_params(params)
        points = self.transform.compute(omegas, translations, points)
        points = self.reshape.compute(points)
        keypoints = self.projection.compute(points)
        keypoints = self.flatten.compute(keypoints)
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


class BundleAdjustment(object):
    def __init__(self, keypoints, camera_parameters):
        """
        keypoints: np.ndarray
            Keypoint coordinates of shape (n_viewpoints, n_points, 2)
        """

        self.initializer = Initializer(keypoints, camera_parameters.matrix)
        n_viewpoints, n_points = keypoints.shape[0:2]
        self.converter = ParameterConverter(n_viewpoints, n_points)

        transformation = Transformer(n_viewpoints, n_points, camera_parameters,
                                     self.converter)
        residual = BaseResidual(keypoints.flatten(), transformation)
        self.optimizer = Optimizer(residual)

    def optimize(self):
        params = self.converter.to_params(*self.initializer.initialize())
        params = self.optimizer.optimize(params)
        return self.converter.from_params(params)
