import numpy as np
from numba import jitclass
import numba


@jitclass([('sigma_squared', numba.float64)])
class GemanMcClure(object):
    def __init__(self, sigma):
        assert(sigma != 0)
        self.sigma_squared = sigma * sigma

    def compute(self, p):
        assert(len(p) == 2)
        x, y = p
        u = x * x + y * y
        return u / (u + self.sigma_squared)
