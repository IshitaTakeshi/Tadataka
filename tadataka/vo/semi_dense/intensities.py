import numpy as np
import numba


@numba.njit
def search_(sequence, kernel):
    min_error = np.inf
    N = len(kernel)
    argmin = None
    for i in range(len(sequence)-N+1):
        d = sequence[i:i+N] - kernel
        error = np.dot(d, d)
        if error < min_error:
            min_error = error
            argmin = i
    return argmin


def search_intensities(intensities_key, intensities_ref):
    argmin = search_(intensities_ref, intensities_key)
    offset = len(intensities_key) // 2
    return argmin + offset
