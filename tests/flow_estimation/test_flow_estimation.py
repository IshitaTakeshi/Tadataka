import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from tadataka.flow_estimation.flow_estimation import estimate_affine_transform

from tests.utils import relative_error


def test_estimate_affine_transform():
    def test(X, A_true, b_true):
        Y = np.dot(A_true, X.T).T + b_true
        transform = estimate_affine_transform(X, Y)

        A_pred = transform.params[0:2, 0:2]
        b_pred = transform.params[0:2, 2]

        assert(relative_error(A_true.flatten(), A_pred.flatten()) < 2e-2)
        assert(relative_error(b_true, b_pred) < 2e-2)

    # random keypoints bofore transformation
    keypoints = np.array([
        [0, 3],
        [1, 4],
        [2, 8],
        [3, 1]
    ])

    A_true = np.array([
        [1, 0],
        [0, 1]
    ])
    b_true = np.array([4, -1])

    test(keypoints, A_true, b_true)

    A_true = np.array([
        [1.2, 0.2],
        [-0.3, 2.0]
    ])
    b_true = np.array([2, -5])

    test(keypoints, A_true, b_true)
