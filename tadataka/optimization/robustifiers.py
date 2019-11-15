from autograd import elementwise_grad
from autograd import numpy as np


class BaseRobustifier(object):
    def robustify(self, x):
        raise NotImplementedError()

    def grad(self, x):
        return elementwise_grad(self.robustify)(x)

    def weights(self, x):
        mask = (x != 0)

        # process only nonzero members of x to avoid division by zero
        y = np.zeros(x.shape)
        y[mask] = self.grad(x[mask]) / x[mask]
        return y


class SquaredRobustifier(BaseRobustifier):
    def robustify(self, x):
        return np.power(x, 2)


class GemanMcClureRobustifier(BaseRobustifier):
    def __init__(self, sigma=0.1):
        self.v = np.power(sigma, 2)

    def robustify(self, x):
        u = np.power(x, 2)
        return u / (u + self.v)
