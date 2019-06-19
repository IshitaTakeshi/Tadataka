from autograd import numpy as np

from optimization.robustifiers import SquaredRobustifier
from optimization.updaters import GaussNewtonUpdater
from optimization.optimizers import BaseOptimizer
from optimization.residuals import Residual
from optimization.transformers import BaseTransformer
from optimization.errors import SumRobustifiedNormError

from rigid.rotation import rodrigues
from rigid.transformation import transform_each


class ParameterConverter(object):
    def __init__(self, n_points, n_viewpoints):
        self.n_points = n_points
        self.n_viewpoints = n_viewpoints

    def from_params(self, params):
        N = 6 * self.n_viewpoints
        pose_params, points = params[:N], params[N:]

        pose_params = pose_params.reshape(-1, 6)
        omegas, translations = pose_params[:, 0:3], pose_params[:, 3:6]

        points = points.reshape(-1, 3)

        return omegas, translations, points

    @property
    def n_dims(self):
        return 6 * self.n_viewpoints + 3 * self.n_points



class Transformer(BaseTransformer):
    def __init__(self, projection, converter):
        # There are no objects to be converted. We just adjust parameters
        super().__init__(None)

        self.projection = projection
        self.converter = converter

    def transform(self, params):
        """
        """
        omegas, translations, points = self.converter.from_params(params)

        rotations = rodrigues(omegas)

        # points.shape == (n_points, n_viewpoints, 3) after transformation
        points = transform_each(rotations, translations, points)

        # points.shape == (n_viewpoints * n_points, 3)
        points = points.reshape(-1, 3)

        return self.projection.project(points)


class BundleAdjustment(object):
    def __init__(self, keypoints, projection, n_points, n_viewpoints):
        self.converter = ParameterConverter(n_points, n_viewpoints)
        self.residual = Residual(Transformer(projection, self.converter), keypoints)

    def optimize(self):
        initial_params = np.ones(self.converter.n_dims)

        robustifier = SquaredRobustifier()
        updater = GaussNewtonUpdater(self.residual, robustifier)
        error = SumRobustifiedNormError(self.residual, robustifier)
        optimizer = BaseOptimizer(updater, error)
        params = optimizer.optimize(initial_params, n_max_iter=1000)
        return self.converter.from_params(params)
