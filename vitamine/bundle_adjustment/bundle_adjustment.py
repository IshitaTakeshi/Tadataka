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
        N = self.converter.n_valid_viewpoints
        M = self.converter.n_valid_points

        self.transform = RigidTransform()
        self.reshape1 = Reshape((N * M, 3))
        self.projection = PerspectiveProjection(camera_parameters)
        self.reshape2 = Reshape((N, M, 2))

    def compute(self, params):
        omegas, translations, points = self.converter.from_params(params)

        points = self.transform.compute(omegas, translations, points)
        points = self.reshape1.compute(points)
        keypoints = self.projection.compute(points)
        keypoints = self.reshape2.compute(keypoints)
        return keypoints


class MaskedResidual(BaseResidual):
    def compute(self, theta):
        residual = super().compute(theta)

        mask = keypoint_mask(residual)

        # ndim of residual will be reduced from 3 to 2 by masking
        residual = residual[mask]
        assert(np.all(~np.isnan(residual)))

        # but explicitly reshape it
        # Because expcilit is better than implicit
        return residual.reshape(-1, 2)


class BundleAdjustmentSolver(object):
    def __init__(self, residual):
        robustifier = SquaredRobustifier()
        updater = GaussNewtonUpdater(residual, robustifier)
        error = SumRobustifiedNormError(robustifier)
        self.optimizer = Optimizer(updater, residual, error)

    def solve(self, initial_params):
        return self.optimizer.optimize(initial_params)


def bundle_adjustment(keypoints, camera_parameters,
                      initial_omegas=None, initial_translations=None,
                      initial_points=None):

    converter = ParameterConverter()

    initializer = Initializer(keypoints, camera_parameters.matrix,
                              initial_omegas, initial_translations,
                              initial_points)

    params = converter.to_params(*initializer.initialize())
    keypoints = converter.mask_keypoints(keypoints)

    transformer = Transformer(camera_parameters, converter)
    residual = MaskedResidual(keypoints, transformer)

    solver = BundleAdjustmentSolver(residual).solve(params)
    return converter.from_params(params)
