from autograd import numpy as np

from vitamine.optimization.functions import Function
from vitamine.transform import AffineTransform
from vitamine.optimization.residuals import BaseResidual
from vitamine.optimization.robustifiers import (
    GemanMcClureRobustifier, SquaredRobustifier)
from vitamine.optimization.updaters import GaussNewtonUpdater
from vitamine.optimization.optimizers import Optimizer
from vitamine.optimization.transformers import BaseTransformer
from vitamine.optimization.errors import SumRobustifiedNormError

# we handle point coordinates P in a format:
# P[:, 0] contains x coordinates
# P[:, 1] contains y coordinates


def theta_to_affine_params(theta):
    A = np.reshape(theta[0:4], (2, 2))
    b = theta[4:6]
    return A, b


class AffineTransformer(Function):
    def __init__(self, keypoints):
        self.keypoints = keypoints

    def compute(self, theta):
        A, b = theta_to_affine_params(theta)
        return AffineTransform(A, b).transform(self.keypoints)


def predict(keypoints1, keypoints2, initial_theta):
    """
    Predict affine transformatin by minimizing the cost function Eq. (2)
    """

    transformer = AffineTransformer(keypoints1)
    residual = BaseResidual(keypoints2, transformer)
    # TODO Geman-McClure is used in the original paper
    robustifier = SquaredRobustifier()
    updater = GaussNewtonUpdater(residual, robustifier)
    error = SumRobustifiedNormError(robustifier)
    optimizer = Optimizer(updater, residual, error)
    return optimizer.optimize(initial_theta, max_iter=1000)


def estimate_affine_transform(keypoints1, keypoints2):
    initial_theta = initialize_theta()
    theta_pred = predict(keypoints1, keypoints2, initial_theta)
    A, b = theta_to_affine_params(theta_pred)
    return A, b
    return AffineTransform(A, b)


def initialize_theta(initial_A=None, initial_b=None):
    if initial_A is None:
        initial_A = np.identity(2)

    if initial_b is None:
        initial_b = np.zeros(2)

    return np.concatenate((
        initial_A.flatten(),
        initial_b.flatten()
    ))
