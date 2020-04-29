import numpy as np
from collections import namedtuple
from tadataka.numeric import safe_invert


Hypothesis = namedtuple("Hypothesis", ["inv_depth", "variance"])


class HypothesisMap(object):
    def __init__(self, inv_depth: np.ndarray, variance: np.ndarray):
        assert(inv_depth.shape == variance.shape)
        self.inv_depth = inv_depth
        self.variance = variance
        self.shape = inv_depth.shape

    @property
    def depth(self):
        return safe_invert(self.inv_depth)
