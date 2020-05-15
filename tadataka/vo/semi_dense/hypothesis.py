import numpy as np
from collections import namedtuple
from tadataka.numeric import safe_invert


Hypothesis = namedtuple("Hypothesis", ["inv_depth", "variance"])


class HypothesisMap(object):
    def __init__(self, inv_depth_map: np.ndarray, variance_map: np.ndarray):
        assert(inv_depth_map.shape == variance_map.shape)
        self.inv_depth_map = inv_depth_map
        self.variance_map = variance_map
        self.shape = inv_depth_map.shape

    @property
    def depth_map(self):
        return safe_invert(self.inv_depth_map)
