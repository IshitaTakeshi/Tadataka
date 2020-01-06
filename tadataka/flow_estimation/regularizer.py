import numpy as np


def get_geman_mcclure(sigma):
    assert(sigma > 0.0)
    sigma_squared = sigma * sigma
    def geman_mcclure(x):
        y = np.sum(np.power(x, 2), axis=1)
        return y / (y + sigma_squared)
    return geman_mcclure
