from autograd import numpy as np

from numpy.testing import assert_array_equal
from vitamine.correspondences import count_correspondences


def test_correspondences():
    points = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [np.nan, np.nan, np.nan],
        [7, 8, 9]
    ])

    keypoints = np.array([
        [[np.nan, np.nan],
         [np.nan, np.nan],
         [np.nan, np.nan],
         [np.nan, np.nan]],
        [[np.nan, np.nan],
         [-1, -2],
         [-3, -4],
         [-5, -6]],
        [[-1, -2],
         [np.nan, np.nan],
         [-3, -4],
         [np.nan, np.nan]]
    ])

    assert_array_equal(
        count_correspondences(points, keypoints),
        np.array([0, 2, 1])
    )
