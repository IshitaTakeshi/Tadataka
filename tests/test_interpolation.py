import numpy as np
from numpy.testing import assert_array_almost_equal
from tadataka.interpolation import interpolate


def test_interpolate():
    # image.shape == (2, 2, 4)

    image = np.array([
        [[0, 0, 0, 0], [1, 0, 1, 0]],
        [[0, 1, 1, 0], [0, 0, 0, 1]]
    ])

    print(image.shape)

    coordinates = np.array([
    #      x    y
        [0.3, 0.6]
    ])
    intensities = interpolate(image, coordinates)

    expected = (image[0, 0] * (1 - 0.3) * (1 - 0.6) +   # x, y = 0, 0
                image[0, 1] * 0.3 * (1 - 0.6) +         # x, y = 1, 0
                image[1, 0] * 0.6 * (1 - 0.3) +         # x, y = 0, 1
                image[1, 1] * 0.3 * 0.6)                # x, y = 1, 1
    assert_array_almost_equal(intensities[0], expected)
