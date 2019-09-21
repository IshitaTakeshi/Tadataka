from autograd import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

from vitamine.matrix import solve_linear


def test_solve_linear():
    # some random matrix
    A = np.array(
        [[7, 3, 6, 7, 4, 3, 7, 2],
         [0, 1, 5, 2, 9, 5, 9, 7],
         [7, 5, 2, 3, 4, 1, 4, 3]]
    )
    x = solve_linear(A)
    assert_equal(x.shape, (8,))
    assert_array_almost_equal(np.dot(A, x), np.zeros(3))
