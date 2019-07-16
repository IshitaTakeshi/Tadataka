from autograd import numpy as np

from scipy.optimize import least_squares


class BaseOptimizer(object):
    def __init__(self, updater, residual, error):
        self.updater = updater
        self.residual = residual
        self.error = error

    def calc_error(self, theta):
        residuals = self.residual.compute(theta)
        return self.error.compute(residuals)

    def optimize(self):
        raise NotImplementedError()


# TODO should be a better name
class Optimizer(BaseOptimizer):
    def optimize(self, initial_theta, max_iter=200):
        theta = initial_theta
        last_error = float('inf')
        for i in range(max_iter):
            d = self.updater.compute(theta)
            print("iteration: {:>8d}  error: {}".format(i, current_error))
            current_error = self.calc_error(theta)

            if current_error >= last_error:
                return theta

            theta = theta - d
            last_error = current_error

        return theta


class ScipyLeastSquaresOptimizer(BaseOptimizer):
    def optimize(self, initial_theta):
        res = least_squares(self.updater.residual, initial_theta,
                            self.updater.jacobian,
                            loss=self.error.compute,
                            ftol=0.1, max_nfev=20, verbose=2)
        return res.x
