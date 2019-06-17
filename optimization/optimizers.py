from autograd import numpy as np


class BaseOptimizer(object):
    def __init__(self, updater, error):
        self.updater = updater
        self.error = error

    def optimize(self, initial_theta, n_max_iter=200):
        theta = initial_theta
        last_error = float('inf')
        for i in range(n_max_iter):
            d = self.updater.compute(theta)
            current_error = self.error.compute(theta)

            if current_error >= last_error:
                return theta

            theta = theta - d
            last_error = current_error

        return theta
