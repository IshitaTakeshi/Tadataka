from numpy.testing import assert_array_equal, assert_array_almost_equal
from vitamine.flow_estimation.flow_estimation import (
    estimate_affine_transform, AffineTransformer, predict,
    affine_params_to_theta, theta_to_affine_params)
from autograd import numpy as np


def test_transformer():
    keypoints = np.array([
        [0, 2],
        [3, 1],
        [2, 4]
    ])

    # A = [[1, 2],
    #      [0.5, 1]]
    # b = [-1, -2]

    theta = np.array([1, 2, 0.5, 1, -1, -2])
    transformer = AffineTransformer(keypoints)
    expected = np.array([
        [3, 0],
        [4, 0.5],
        [9, 3]
    ])
    assert_array_equal(transformer.compute(theta), expected)

    from autograd import jacobian
    J = jacobian(transformer.compute)(theta)
    print("J")
    print(J)



def test_estimate_affine_transform():
    def test(keypoints1, A_true, b_true):
        keypoints2 = np.dot(A_true, keypoints1.T).T + b_true
        A_pred, b_pred = estimate_affine_transform(keypoints1, keypoints2)
        assert_array_almost_equal(A_true, A_pred)
        assert_array_almost_equal(b_true, b_pred)

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
        [1, 2],
        [-0.4, 3]
    ])
    b_true = np.array([0, 0])

    test(keypoints, A_true, b_true)

    A_true = np.array([
        [1.2, 0.2],
        [-0.3, 2.0]
    ])
    b_true = np.array([8, -2])

    test(keypoints, A_true, b_true)


def test_affine_params_to_theta():
    A = np.array([[0, 1],
                  [2, 3]])
    b = np.array([4, 5])
    theta = affine_params_to_theta(A, b)
    assert_array_equal(theta, np.arange(6))


def test_theta_to_affine_params():
    A, b = theta_to_affine_params(np.arange(6))

    assert_array_equal(A,
                       np.array([[0, 1],
                                 [2, 3]]))

    assert_array_equal(b, np.array([4, 5]))
