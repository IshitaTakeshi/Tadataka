from autograd import numpy as np
from numpy.testing import assert_array_equal

from vitamine.feature_matching import MatchMatrixGenerator


def test_match_matrix_generator():
    generator = MatchMatrixGenerator()

    generator.add(0, 1,
                  np.array([[0, 0],
                            [1, 2]]))

    generator.add(0, 2,
                  np.array([[1, 0],
                            [3, 1]]))

    generator.add(0, 3,
                  np.array([[3, 0],
                            [2, 3]]))

    generator.add(1, 2,
                  np.array([[1, 2]]))

    generator.add(1, 3,
                  np.array([[2, 1],
                            [1, 2]]))

    generator.add(2, 3,
                  np.array([[4, 3]]))

    assert_array_equal(
        generator.matrix(),
        np.array([
            [0, 0, np.nan, np.nan],
            [1, 2, 0, 1],
            [3, np.nan, 1, 0],
            [2, np.nan, 4, 3],
            [np.nan, 1, 2, 2]
        ]).T
    )
