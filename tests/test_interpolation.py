import numpy as np
from numpy.testing import assert_almost_equal

import pytest

from tadataka.interpolation import interpolation


def test_interpolation():
    image = np.array([
        [0, 1, 5],
        [0, 0, 2],
        [4, 3, 2],
        [5, 6, 1]
    ], dtype=np.float64)
    # width, height = (3, 4)

    coordinates = np.array([
    #      x    y
        [0.0, 0.0],
        [1.3, 2.6],  # ordinary
        [2.0, 2.9],  # maximum x
        [1.9, 3.0],  # maximum y
        [2.0, 3.0],  # possible maximum coordinate
    ])
    intensities = interpolation(image, coordinates)

    assert_almost_equal(intensities[0], image[0, 0])

    #                  y  x
    expected1 = (image[2, 1] * (2.0 - 1.3) * (3.0 - 2.6) +
                 image[2, 2] * (1.3 - 1.0) * (3.0 - 2.6) +
                 image[3, 1] * (2.0 - 1.3) * (2.6 - 2.0) +
                 image[3, 2] * (1.3 - 1.0) * (2.6 - 2.0))
    assert_almost_equal(intensities[1], expected1)

    #                  y  x
    expected2 = (image[2, 2] * (3.0 - 2.0) * (3.0 - 2.9) +
                 image[3, 2] * (3.0 - 2.0) * (2.9 - 2.0))
    assert_almost_equal(intensities[2], expected2)

    #                  y  x
    expected3 = (image[3, 1] * (2.0 - 1.9) * (4.0 - 3.0) +
                 image[3, 2] * (1.9 - 1.0) * (4.0 - 3.0))
    assert_almost_equal(intensities[3], expected3)

    assert_almost_equal(intensities[4], image[3, 2])

    # 1d coordinate input
    intensity = interpolation(image, coordinates[2])
    assert_almost_equal(intensity, expected2)

    with pytest.raises(ValueError):
        interpolation(image, [3.0, 2.01])

    with pytest.raises(ValueError):
        interpolation(image, [3.01, 2.0])

    with pytest.raises(ValueError):
        interpolation(image, [-0.01, 0.0])

    with pytest.raises(ValueError):
        interpolation(image, [0.0, -0.01])
