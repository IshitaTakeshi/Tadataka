from autograd import numpy as np
from numpy.testing import assert_array_equal, assert_equal

from vitamine.bundle_adjustment.parameters import (
    ParameterMask, from_params, to_params)


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

keypoints_true = np.array([
    [[1, 2],
     [3, 4],
     [5, 6],
     [7, 8]],
    [[-1, -2],
     [-3, -4],
     [-5, -6],
     [-7, -8]],
])

mask = ParameterMask(omegas_true, translations_true, points_true)


def test_n_valid_viewpoints():
    assert_equal(mask.n_valid_viewpoints, 1)


def test_n_valid_points():
    assert_equal(mask.n_valid_points, 3)


def test_get_masked():
    omegas, translations, points = mask.get_masked()

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


def test_mask_keypoints():
    assert_array_equal(
        mask.mask_keypoints(keypoints_true),
        np.array([
            [[-1, -2],
             [-3, -4],
             [-7, -8]],
        ])
    )


def test_fill():
    omegas = np.array([[4, 5, 6]])
    translations = np.array([[10, 11, 12]])
    points = np.array([
        [-1, -2, -3],
        [-4, -5, -6],
        [-10, -11, -12]
    ])
    omegas, translations, points = mask.fill(omegas, translations, points)

    assert_array_equal(
        omegas,
        np.array([
            [np.nan, np.nan, np.nan],
            [4, 5, 6],
        ])
    )

    assert_array_equal(
        translations,
        np.array([
            [np.nan, np.nan, np.nan],
            [10, 11, 12]
        ])
    )

    assert_array_equal(
        points,
        np.array([
            [-1, -2, -3],
            [-4, -5, -6],
            [np.nan, np.nan, np.nan],
            [-10, -11, -12]
        ])
    )

def test_to_params():
    omegas = np.array([[4, 5, 6]])
    translations = np.array([[10, 11, 12]])
    points = np.array([
        [-1, -2, -3],
        [-4, -5, -6],
        [-10, -11, -12]
    ])

    assert_array_equal(
        to_params(omegas, translations, points),
        np.array([
            4, 5, 6,
            10, 11, 12,
            -1, -2, -3,
            -4, -5, -6,
            -10, -11, -12
        ])
    )


def test_from_params():
    params = np.array([
        4, 5, 6,
        10, 11, 12,
        -1, -2, -3,
        -4, -5, -6,
        -10, -11, -12
    ])

    omegas, translations, points =\
        from_params(params, mask.n_valid_viewpoints, mask.n_valid_points)

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
