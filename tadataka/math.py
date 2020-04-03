from numba import njit


@njit
def weighted_mean(x, w):
    # numpy.average can do the same computation
    # but we need this function to accelerate with numba.njit
    assert(x.shape == w.shape)

    s = w.sum()
    if s == 0:
        raise ValueError("Sum of weights is zero")

    return (x * w).sum() / s
