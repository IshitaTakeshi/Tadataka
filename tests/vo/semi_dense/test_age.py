from numpy.testing import assert_array_equal
import numpy as np

from tadataka.vo.semi_dense.age import increment_


def test_increment_():
    age0 = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ])
    us0 = np.array([
    #    x  y
        [0, 1],
        [1, 2],
        [2, 2]
    ])
    us1 = np.array([
    #    x  y
        [0, 0],
        [1, 2],
        [2, 0]
    ])

    age1 = increment_(age0, us0, us1)
    assert_array_equal(
        age1,
        [[4, 0, 9],
         [0, 0, 0],
         [0, 8, 0]]
    )
