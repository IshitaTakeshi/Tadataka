import numpy as np
from numpy.testing import assert_almost_equal

from tadataka.math import weighted_mean


def test_weighted_mean():
    x = np.array([2.0, 3.0])
    w = np.array([0.9, 0.1])
    assert_almost_equal(weighted_mean(x, w), 1.8 + 0.3)

    x = np.array([2.0, 3.0])
    w = np.array([0.4, 0.8])
    assert_almost_equal(weighted_mean(x, w), (0.8 + 2.4) / 1.2)
