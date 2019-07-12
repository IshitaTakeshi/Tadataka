from autograd import jacobian
from autograd import numpy as np


class GradientBasedUpdater(object):
    def flattened_residual(self, theta):
        residual = self.residual.compute(theta)
        return residual.flatten()

    def jacobian(self, theta):
        return jacobian(self.residual.compute)(theta)


class GaussNewtonUpdater(GradientBasedUpdater):
    def __init__(self, residual, robustifier):
        self.residual = residual
        self.robustifier = robustifier

    def compute(self, theta):
        # Not exactly the same as the equation of Gauss-Newton update
        # d = inv (J^T * J) * J * r
        # however, it works better than implementing the equation malually

        r = self.flattened_residual(theta)
        J = self.jacobian(theta)

        assert(np.ndim(r) == 1)

        # residuals can be a multi-dimensonal array so flatten them
        J = J.reshape(r.shape[0], theta.shape[0])

        # TODO add weighted Gauss-Newton as an option
        # weights = self.robustifier.weights(r)
        delta, error, _, _ = np.linalg.lstsq(J, r, rcond=None)
        return delta
