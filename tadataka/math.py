import numpy as np
from scipy.sparse import linalg


def weighted_mean(x, w):
    # numpy.average can do the same computation
    assert(x.shape == w.shape)

    s = w.sum()
    if s == 0:
        raise ValueError("Sum of weights is zero")

    return (x * w).sum() / s


def get_solver_(method, **kwargs):
    def lstsq(A, b):
        x, _, _, _ = np.linalg.lstsq(A, b, **kwargs)
        return x

    def cg(A, b):
        x, _ = linalg.cg(np.dot(A.T, A), np.dot(A.T, b), **kwargs)
        return x

    if method == "lstsq":
        return lstsq

    if method == "cg":
        return cg


def solve_linear_equation(A, b, weights=None, method="lstsq", **kwargs):
    solve = get_solver_(method)

    assert(A.shape[0] == b.shape[0])

    if weights is None:
        return solve(A, b)

    assert(A.shape[0] == weights.shape[0])

    w = np.sqrt(weights)
    b = b * w
    A = A * w.reshape(-1, 1)
    return solve(A, b)
