from autograd import numpy as np

from vitamine.bundle_adjustment.initializers import Initializer
from vitamine.bundle_adjustment.parameters import ParameterConverter
from vitamine.bundle_adjustment.mask import keypoint_mask, point_mask

from vitamine.optimization.robustifiers import SquaredRobustifier
from vitamine.optimization.updaters import GaussNewtonUpdater
from vitamine.optimization.array_utils import Flatten, Reshape
from vitamine.optimization.transformers import BaseTransformer
from vitamine.optimization.errors import BaseError, SumRobustifiedNormError
from vitamine.optimization.functions import Function
from vitamine.optimization.residuals import BaseResidual
from vitamine.optimization.optimizers import Optimizer
from vitamine.optimization.transformers import BaseTransformer

from vitamine.projection.projections import PerspectiveProjection

from vitamine.rigid.rotation import rodrigues
from vitamine.rigid.transformation import transform_each


class RigidTransform(Function):
    def compute(self, omegas, translations, points):
        return transform_each(rodrigues(omegas), translations, points)


class Transformer(BaseTransformer):
    def __init__(self, camera_parameters, converter):
        self.converter = converter

        self.transform = RigidTransform()
        self.projection = PerspectiveProjection(camera_parameters)

    def compute(self, params):
        omegas, translations, points = self.converter.from_params(params)

        N = self.converter.n_valid_viewpoints
        M = self.converter.n_valid_points

        points = self.transform.compute(omegas, translations, points)

        points = points.reshape((N * M, 3))

        keypoints = self.projection.compute(points)

        keypoints = keypoints.reshape((N, M, 2))

        return keypoints


class MaskedResidual(BaseResidual):
    def __init__(self, y, transformer, converter):
        super().__init__(y, transformer)
        self.converter = converter

    def compute(self, theta):
        x = self.transformer.compute(theta)

        # FIXME just not wise
        y = self.y[self.converter.pose_mask]
        y = y[:, self.converter.point_mask]

        residual = y - x
        mask = keypoint_mask(residual)

        # ndim of residual will be reduced from 3 to 2
        residual = residual[mask]
        # but explicitly reshape it
        residual = residual.reshape(-1, 2)

        assert(np.all(~np.isnan(residual)))
        return residual


class BundleAdjustmentSolver(object):
    def __init__(self, residual):
        robustifier = SquaredRobustifier()
        updater = GaussNewtonUpdater(residual, robustifier)
        error = SumRobustifiedNormError(robustifier)
        self.optimizer = Optimizer(updater, residual, error)

    def solve(self, initial_params):
        return self.optimizer.optimize(initial_params)


class BundleAdjustment(object):
    def __init__(self, keypoints, camera_parameters,
                 initial_omegas=None, initial_translations=None,
                 initial_points=None):
        """
        keypoints: np.ndarray
            Keypoint coordinates of shape (n_viewpoints, n_points, 2)
        """

        self.initializer = Initializer(keypoints, camera_parameters.matrix,
                                       initial_omegas, initial_translations,
                                       initial_points)

        self.converter = ParameterConverter()
        transformer = Transformer(camera_parameters, self.converter)
        residual = MaskedResidual(keypoints, transformer, self.converter)

        self.solver = BundleAdjustmentSolver(residual)

    def optimize(self):
        params = self.initializer.initialize()
        params = self.converter.to_params(*params)
        assert(np.all(~np.isnan(params)))
        params = self.solver.solve(params)
        return self.converter.from_params(params)
