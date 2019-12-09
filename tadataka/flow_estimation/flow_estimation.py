from autograd import numpy as np

from tadataka.optimization.functions import Function
from tadataka.transform import AffineTransform
from tadataka.optimization.residuals import BaseResidual
from tadataka.optimization.robustifiers import (
    GemanMcClureRobustifier, SquaredRobustifier)
from tadataka.optimization.updaters import GaussNewtonUpdater
from tadataka.optimization.optimizers import Optimizer
from tadataka.optimization.transformers import BaseTransformer
from tadataka.optimization.errors import SumRobustifiedNormError
from tadataka import irls
# we handle point coordinates P in a format:
# P[:, 0] contains x coordinates
# P[:, 1] contains y coordinates


def affine_params_to_theta(A, b):
    return np.concatenate((A.flatten(), b.flatten()))


def theta_to_affine_params(theta):
    A = np.reshape(theta[0:4], (2, 2))
    b = theta[4:6]
    return A, b


def estimate_affine_transform(keypoints1, keypoints2):
    keypoints1 = np.column_stack((keypoints1, np.ones(keypoints1.shape[0])))
    params0 = irls.fit(keypoints1, keypoints2[:, 0])
    params1 = irls.fit(keypoints1, keypoints2[:, 1])

    A0, b0 = params0[0:2], params0[2]
    A1, b1 = params1[0:2], params1[2]

    A = np.vstack((A0, A1))
    b = np.array([b0, b1])
    return AffineTransform(A, b)
