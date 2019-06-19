from autograd import numpy as np


class BaseError(object):
    def compute(self):
        raise NotImplementedError()


class SumRobustifiedNormError(BaseError):
    def __init__(self, residual, robustifier):
        self.residual = residual
        self.robustifier = robustifier

    def compute(self, theta):
        r = self.residual.residuals(theta)
        norms = np.linalg.norm(r, axis=1)
        return np.sum(self.robustifier.robustify(norms))
