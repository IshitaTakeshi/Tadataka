import numpy as np

from tadataka.optimization.functions import Function


class BaseError(Function):
    def compute(self, residual):
        raise NotImplementedError()


class SumRobustifiedNormError(BaseError):
    def __init__(self, robustifier):
        self.robustifier = robustifier

    def compute(self, residuals):
        norms = np.linalg.norm(residuals, axis=1)
        return np.sum(self.robustifier.robustify(norms))
