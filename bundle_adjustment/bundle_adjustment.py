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
    def __init__(self, n_viewpoints, n_points):
        self.n_viewpoints = n_viewpoints
        self.n_points = n_points

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
        omegas, translations, points = self.converter.from_params(params)

        rotations = rodrigues(omegas)

        # points.shape == (n_viewpoints, n_points, 3) after transformation
        points = transform_each(rotations, translations, points)

        shape = points.shape[0:2]
        points = points.reshape(-1, 3)  # flatten once

        keypoints = self.projection.project(points)  # project

        return keypoints.reshape(*shape, 2)  # restore the shape


class BundleAdjustment(object):
    def __init__(self, keypoints, projection):
        """
        keypoints: np.ndarray
            Keypoint coordinates of shape (n_viewpoints, n_points, 2)
        projection: projection model
        """
        n_viewpoints, n_points = keypoints.shape[0:2]
        self.converter = ParameterConverter(n_viewpoints, n_points)

        transformer = Transformer(projection, self.converter)
        self.residual = Residual(transformer, keypoints)

    def optimize(self):
        initial_params = np.ones(self.converter.n_dims)

        robustifier = SquaredRobustifier()
        updater = GaussNewtonUpdater(self.residual, robustifier)
        error = SumRobustifiedNormError(self.residual, robustifier)
        optimizer = BaseOptimizer(updater, error)
        params = optimizer.optimize(initial_params, n_max_iter=1000)
        return self.converter.from_params(params)
