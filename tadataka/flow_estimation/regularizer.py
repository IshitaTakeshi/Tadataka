import numpy as np


def get_geman_mcclure(sigma):
    assert(sigma > 0.0)
    def geman_mcclure(x):
        y = np.sum(np.power(x, 2))
        return y / (y + sigma)
    return geman_mcclure
