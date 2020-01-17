import numpy as np
from numpy.testing import assert_array_almost_equal
from tadataka.interpolate import interpolate


def test_interpolate():
    # image.shape == (2, 2, 4), 4 channels 2 x 2 image

    image = np.array([
        [[0, 0, 0, 0], [1, 0, 1, 0]],
        [[0, 1, 1, 0], [0, 0, 0, 1]]
    ])

    coordinates = np.array([
    #      x    y
        [0.3, 0.6],
        [0.1, 0.8]
    ])
    intensities = interpolate(image, coordinates)

    expected = (image[0, 0] * (1 - 0.3) * (1 - 0.6) +   # x, y = 0, 0
                image[0, 1] * 0.3 * (1 - 0.6) +         # x, y = 1, 0
                image[1, 0] * 0.6 * (1 - 0.3) +         # x, y = 0, 1
                image[1, 1] * 0.3 * 0.6)                # x, y = 1, 1
    assert_array_almost_equal(intensities[0], expected)

    expected = (image[0, 0] * (1 - 0.1) * (1 - 0.8) +   # x, y = 0, 0
                image[0, 1] * 0.1 * (1 - 0.8) +         # x, y = 1, 0
                image[1, 0] * 0.8 * (1 - 0.1) +         # x, y = 0, 1
                image[1, 1] * 0.1 * 0.8)                # x, y = 1, 1
    assert_array_almost_equal(intensities[1], expected)
