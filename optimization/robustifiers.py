from autograd import elementwise_grad
from autograd import numpy as np


class Robustifier(object):
    def robustify(self, x):
        raise NotImplementedError()

    def grad(self, x):
        return elementwise_grad(self.robustify)(x)

    def weights(self, x):
        mask = (x != 0)

        # process only nonzero members to avoid division by zero
        # for members where x == 0, we set 0 to corresponding y
        # because usually x = norm(residual), so x == 0 means that
        # residual is zero and weighting is not required

        y = np.zeros(x.shape)
        y[mask] = self.grad(x[mask]) / x[mask]
        return y


class SquaredRobustifier(Robustifier):
    def robustify(self, x):
        return np.power(x, 2)


class GemanMcClureRobustifier(Robustifier):
    def __init__(self, sigma=0.1):
        self.v = np.power(sigma, 2)

    def robustify(self, x):
        u = np.power(x, 2)
        return u / (u + self.v)
