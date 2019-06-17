from autograd import elementwise_grad
from autograd import numpy as np


class Robustifier(object):
    def robustify(self, x):
        raise NotImplementedError()

    def grad(self, x):
        return elementwise_grad(self.robustify)

    def weights(self, x):
        return self.grad(norms) / norms


class SquaredRobustifier(Robustifier):
    def robustify(self, x):
        return np.power(x, 2)


class GemanMcClureRobustifier(Robustifier):
    def __init__(self, sigma=0.1):
        self.v = np.power(sigma, 2)

    def robustify(self, x):
        u = np.power(x, 2)
        return u / (u + self.v)
