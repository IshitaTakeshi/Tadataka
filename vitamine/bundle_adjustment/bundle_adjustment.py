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

from vitamine.projection.projections import PerspectiveProjection

from vitamine.rigid.rotation import rodrigues
from vitamine.rigid.transformation import transform_each


class RigidTransform(Function):
    def compute(self, omegas, translations, points):
        return transform_each(rodrigues(omegas), translations, points)


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
    def __init__(self, robustifier):
        super().__init__()

        self.reshape = Reshape((-1, 2))
        self.error = SumRobustifiedNormError(robustifier)

    def compute(self, residual):
        residual = self.reshape.compute(residual)
        return self.error.compute(residual)


class MaskedResidual(BaseResidual):
    def __init__(self, y, transformer, masks):
        super().__init__(y, transformer)
        self.masks = masks

    def compute(self, theta):
        residual = super().compute(theta)
        residual = residual[self.masks].flatten()
        return residual


class BundleAdjustmentSolver(object):
    def __init__(self, residual):
        robustifier = SquaredRobustifier()
        updater = GaussNewtonUpdater(residual, robustifier)
        error = Error(residual, robustifier)
        self.optimizer = Optimizer(updater, error)

    def solve(self, initial_params):
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

        transformer = Transformer(n_viewpoints, n_points, camera_parameters,
                                  self.converter)
        residual = MaskedResidual(keypoints, transformer,
                                  keypoint_mask(keypoints))
        self.solver = BundleAdjustmentSolver(residual)

    def optimize(self):
        params = self.initializer.initialize()
        params = self.converter.to_params(*params)
        assert(np.all(~np.isnan(params)))
        params = self.solver.solve(params)
        return self.converter.from_params(params)
