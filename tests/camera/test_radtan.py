import numpy as np

from numpy.testing import assert_array_almost_equal
from tadataka.camera._radtan import inv2x2


def test_inv2x2():
    X = np.array([
        [100, 2],
        [40, -30]
    ], dtype=np.float64)
    assert_array_almost_equal(np.dot(inv2x2(X), X), np.identity(2))
