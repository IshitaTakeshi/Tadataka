from autograd import numpy as np
from numpy.testing import assert_array_equal
from bundle_adjustment.bundle_adjustment import ParameterConverter


def test_parameter_converter():
    n_viewpoints = 2
    n_points = 4

    params = np.array([
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
        -1, -2, -3,
        -4, -5, -6,
        -7, -8, -9,
        -10, -11, -12
    ])

    converter = ParameterConverter(n_viewpoints, n_points)
    omegas, translations, points = converter.from_params(params)

    assert_array_equal(
        points,
        np.array([
            [-1, -2, -3],
            [-4, -5, -6],
            [-7, -8, -9],
            [-10, -11, -12]
        ])
    )

    assert_array_equal(
        omegas,
        np.array([
            [1, 2, 3],
            [7, 8, 9]
        ])
    )

    assert_array_equal(
        translations,
        np.array([
            [4, 5, 6],
            [10, 11, 12]
        ])
    )
