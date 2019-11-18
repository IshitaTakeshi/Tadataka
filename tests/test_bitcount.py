import numpy as np
from numpy.testing import assert_array_equal

from tadataka import bitcount


D = bitcount.distances(
    np.array([[1, 0, 0, 1],
              [1, 0, 1, 1]], dtype=np.bool),
    np.array([[1, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 1, 1, 0]], dtype=np.bool)
)

assert_array_equal(
    D,
    np.array([
        [1, 1, 4],
        [2, 0, 3]
    ])
)


D = bitcount.distances(
    np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
              [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]], dtype=np.bool),
    np.array([[1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1],
              [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
              [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]], dtype=np.bool)
)

assert_array_equal(
    D,
    np.array([
        [4, 5, 10],
        [8, 5, 6]
    ])
)
