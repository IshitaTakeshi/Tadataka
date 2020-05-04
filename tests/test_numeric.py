from numpy.testing import assert_almost_equal
from tadataka.numeric import safe_invert


def test_safe_invert():
    assert_almost_equal(safe_invert(10.0, epsilon=1e-17), 0.1)
    assert_almost_equal(safe_invert(0.0, epsilon=1e-17), 1e17)
