from autograd import numpy as np
from numpy.testing import assert_array_almost_equal
from vitamine.stat import zca_whitening, normalize_mean, ChiSquaredTest


def test_chi_squared_test():
    X = np.random.normal(size=(100000, 2))
    mask = ChiSquaredTest(p=0.95, dof=2).test(X)
    # 95% of elemest should be classified as inliers
    assert_array_almost_equal(np.sum(mask) / len(mask), 0.95, decimal=3)

    X = np.random.normal(size=(100000, 2))
    mask = ChiSquaredTest(p=0.98, dof=2).test(X)
    # 98% of elemest should be classified as inliers
    assert_array_almost_equal(np.sum(mask) / len(mask), 0.98, decimal=3)

    # samples are normalized even
    # if they follow a non-standard normal distribution
    X = np.random.multivariate_normal(mean=[-1.0, 2.0],
                                      cov=[[2.0, 0.5],
                                           [0.5, 2.0]],
                                      size=100000)
    mask = ChiSquaredTest(p=0.95, dof=2).test(X)
    # 95% of elemest should be classified as inliers
    assert_array_almost_equal(np.sum(mask) / len(mask), 0.95, decimal=3)

def test_normalize_mean():
    X = np.random.uniform(-10, 10, (20, 3))
    assert_array_almost_equal(np.mean(normalize_mean(X), axis=0), np.zeros(3))


def test_zca_whitening():
    X = np.random.uniform(-10, 10, (100, 3))
    X = normalize_mean(X)
    Y = zca_whitening(X)

    C = np.cov(Y, rowvar=False)
    assert(np.isclose(C, np.identity(3)).all())
    assert_array_almost_equal(np.mean(Y, axis=0), np.zeros(3))
