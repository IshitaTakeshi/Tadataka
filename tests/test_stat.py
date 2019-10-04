from autograd import numpy as np
from numpy.testing import assert_array_almost_equal
from vitamine.stat import zca_whitening, normalize_mean


def test_chi_squared_test():
    X = np.array([
        [-1, 0, 1],
        [0, 2, 4]
    ])

    mean = np.mean(X, axis=0, keepdims=True)
    Y = X - mean

    for y in Y:
        print(np.dot(y, y))
    # print(np.power(x - m, 2))
    # print(np.sum(np.power(X - mean, 2), axis=1))
    C = np.cov(X)
    print(C)



def test_zca_whitening():
    X = np.random.uniform(-10, 10, (100, 3))
    X = normalize_mean(X)
    Y = zca_whitening(X)

    C = np.cov(Y, rowvar=False)
    assert(np.isclose(C, np.identity(3)).all())
    assert_array_almost_equal(np.mean(Y, axis=0), np.zeros(3))
