import numpy as np
from numpy.testing import assert_array_equal
from tadataka.dataset.match import match_timestamps


def test_match_timestamps():
    #                         0    1    2    3
    timestamps0 = np.array([0.0, 1.0, 2.0, 3.0])
    #                         0    1    2    3    4    5
    timestamps1 = np.array([0.0, 0.2, 1.0, 1.3, 2.1, 3.3])

    matches01 = match_timestamps(
        timestamps0,
        timestamps1,
        max_difference=np.inf
    )

    assert_array_equal(
        matches01,
        [[0, 0],
         [1, 2],
         [2, 4],
         [3, 5]]
    )

    #                         0    1    2    3    4    5
    timestamps0 = np.array([0.0, 0.3, 1.0, 1.8, 2.0, 3.0])
    #                         0    1    2    3
    timestamps1 = np.array([0.0, 1.0, 2.0, 3.0])

    matches01 = match_timestamps(
        timestamps0,
        timestamps1,
        max_difference=np.inf
    )

    assert_array_equal(
        matches01,
        [[0, 0],
         [2, 1],
         [4, 2],
         [5, 3]]
    )
