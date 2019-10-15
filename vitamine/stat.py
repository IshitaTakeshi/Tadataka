from autograd import numpy as np
from scipy.stats import chi2

EPSILON = 1e-10


def normalize_mean(X):
    mean = np.mean(X, axis=0, keepdims=True)
    return X - mean


def zca_whitening(X, normalize_mean=True):
    C = np.cov(X, rowvar=False)
    U, s, V = np.linalg.svd(C)
    S = np.diag(1 / (np.sqrt(s) + EPSILON))
    ZCA = np.dot(U, np.dot(S, U.T))
    return np.dot(ZCA, X.T).T


class ChiSquaredTest(object):
    def __init__(self, p=0.95, dof=2):
        # if p = 0.95, 95% of data samples are regarded as inliers
        # in the assumption that the samples follow
        # the standard normal distribution
        self.threshold = chi2.ppf(p, dof)

    def test(self, X):
        Y = zca_whitening(normalize_mean(X))
        # Y follows the standard normal distribution
        E = np.sum(np.power(Y, 2), axis=1)
        # E follows the chi-squared distribution
        return E <= self.threshold
