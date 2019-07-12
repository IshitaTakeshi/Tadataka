from autograd import numpy as np

from vitamine.optimization.functions import Function


class BaseError(Function):
    def compute(self, residual):
        raise NotImplementedError()


class SumRobustifiedNormError(BaseError):
    def __init__(self, robustifier):
        self.robustifier = robustifier

    def compute(self, residual):
        norms = np.linalg.norm(residual, axis=1)
        return np.sum(self.robustifier.robustify(norms))
