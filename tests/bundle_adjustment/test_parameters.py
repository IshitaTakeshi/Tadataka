from autograd import numpy as np
from numpy.testing import assert_array_equal

from vitamine.bundle_adjustment.parameters import ParameterConverter


def test_parameter_converter():
    # ParameterConverter instance holds states so we test multiple
    # methods in one test function

    omegas_true = np.array([
        [1, np.nan, 3],
        [4, 5, 6],
    ])

    translations_true = np.array([
        [7, 8, 9],
        [10, 11, 12]
    ])

    points_true = np.array([
        [-1, -2, -3],
        [-4, -5, -6],
        [np.nan, -8, np.nan],
        [-10, -11, -12]
    ])

    converter = ParameterConverter()

    #-------------------------------------------------------------------
    # Test 'to_params'
    #-------------------------------------------------------------------

    params = converter.to_params(omegas_true, translations_true, points_true)
    expected = np.array([
        4, 5, 6,
        10, 11, 12,
        -1, -2, -3,
        -4, -5, -6,
        -10, -11, -12
    ])

    assert_array_equal(params, expected)

    #-------------------------------------------------------------------
    # Test 'mask_keypoints'
    #-------------------------------------------------------------------
    keypoints = np.array([
        [[1, 2],
         [3, 4],
         [5, 6],
         [7, 8]],
        [[-1, -2],
         [-3, -4],
         [-5, -6],
         [-7, -8]],
    ])

    expected = np.array([
        [[-1, -2],
         [-3, -4],
         [-7, -8]],
    ])

    assert_array_equal(converter.mask_keypoints(keypoints), expected)

    #-------------------------------------------------------------------
    # Test 'from_params'
    #-------------------------------------------------------------------

    omegas, translations, points = converter.from_params(params)

    assert_array_equal(
        omegas,
        np.array([[4, 5, 6]])
    )

    assert_array_equal(
        translations,
        np.array([[10, 11, 12]])
    )

    assert_array_equal(
        points,
        np.array([
            [-1, -2, -3],
            [-4, -5, -6],
            [-10, -11, -12]
        ])
    )
