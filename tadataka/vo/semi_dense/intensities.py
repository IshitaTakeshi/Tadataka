import numpy as np


def calc_error(v1, v2):
    d = (v2 - v1).flatten()
    return np.dot(d, d)


# TODO use numba for acceleration
def convolve(a, b, error_func):
    # reverese b because the second argument is reversed in the computation
    N = len(b)
    return np.array([error_func(a[i:i+N], b) for i in range(len(a)-N+1)])


def search_intensities(intensities_key, intensities_ref):
    intensities_key = intensities_key
    offset = len(intensities_key) // 2
    errors = convolve(intensities_ref, intensities_key, calc_error)
    argmin = np.argmin(errors)
    return argmin + offset
