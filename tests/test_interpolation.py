import numpy as np
from numpy.testing import assert_almost_equal
from tadataka.interpolation import interpolation


def test_interpolation():
    # image.shape == (2, 2), 4 channels 2 x 2 image

    image = np.array([
        [0, 1, 5],
        [0, 0, 2],
        [4, 3, 2]
    ], dtype=np.float64)

    coordinates = np.array([
    #      x    y
        [0.3, 0.6],
        [0.1, 0.8],
        [1.4, 0.2],
        [1.2, 4.8]
    ])
    intensities = interpolation(image, coordinates, order=1)

    #                 y  x
    expected0 = (image[0, 0] * (1.0 - 0.3) * (1.0 - 0.6) +
                 image[0, 1] * (0.3 - 0.0) * (1.0 - 0.6) +
                 image[1, 0] * (0.6 - 0.0) * (1.0 - 0.3) +
                 image[1, 1] * (0.3 - 0.0) * (0.6 - 0.0))
    assert_almost_equal(intensities[0], expected0)

    #                 y  x
    expected1 = (image[0, 0] * (1.0 - 0.1) * (1.0 - 0.8) +
                 image[0, 1] * (0.1 - 0.0) * (1.0 - 0.8) +
                 image[1, 0] * (0.8 - 0.0) * (1.0 - 0.1) +
                 image[1, 1] * (0.1 - 0.0) * (0.8 - 0.0))
    assert_almost_equal(intensities[1], expected1)

    #                 y  x
    expected2 = (image[0, 1] * (2.0 - 1.4) * (1.0 - 0.2) +
                 image[0, 2] * (1.4 - 1.0) * (1.0 - 0.2) +
                 image[1, 1] * (0.2 - 0.0) * (2.0 - 1.4) +
                 image[1, 2] * (1.4 - 1.0) * (0.2 - 0.0))

    assert_almost_equal(intensities[2], expected2)

    assert(np.isnan(intensities[3]))

    intensity = interpolation(image, np.array([1.4, 0.2]), order=1)
    assert_almost_equal(intensity, expected2)
