from autograd import jacobian
from autograd import numpy as np


class GradientBasedUpdater(object):
    def jacobian(self, theta):
        return jacobian(self.residual.residuals)(theta)


class GaussNewtonUpdater(GradientBasedUpdater):
    def __init__(self, residual, robustifier):
        self.residual = residual
        self.robustifier = robustifier

    def compute(self, theta):
        # Not exactly the same as the equation of Gauss-Newton update
        # d = inv (J^T * J) * J * r
        # however, it works better than implementing the equation malually

        r = self.residual.residuals(theta)
        J = self.jacobian(theta)

        # TODO add weighted Gauss-Newton as an option
        # weights = self.robustifier.weights(r)
        theta, error, _, _ = np.linalg.lstsq(J, r, rcond=None)
        return theta
