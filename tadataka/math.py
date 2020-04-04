from numba import njit
import numpy as np


@njit
def weighted_mean(x, w):
    # numpy.average can do the same computation
    # but we need this function to accelerate with numba.njit
    assert(x.shape == w.shape)

    s = w.sum()
    if s == 0:
        raise ValueError("Sum of weights is zero")

    return (x * w).sum() / s


@njit(fastmath=True)
def solve_linear_equation(A, b, weights=None):
    assert(A.shape[0] == b.shape[0])

    if weights is None:
        return np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)

    assert(A.shape[0] == weights.shape[0])

    ATW = A.T * weights
    return np.linalg.inv(ATW.dot(A)).dot(ATW).dot(b)
