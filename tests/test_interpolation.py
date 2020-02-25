import numpy as np
from numpy.testing import assert_almost_equal
from tadataka.interpolation import interpolation


def test_interpolation():
    # image.shape == (2, 2), 4 channels 2 x 2 image

    image = np.array([
        [0, 1, 5],
        [0, 0, 2],
        [4, 3, 2],
        [5, 6, 1]
    ], dtype=np.float64)

    coordinates = np.array([
    #      x    y
        [0.3, 0.6],
        [0.1, 0.8],
        [0.4, 2.2],
        [1.0, 2.0]
    ])
    intensities = interpolation(image, coordinates)

    #                  y  x
    expected0 = (image[0, 0] * (1.0 - 0.3) * (1.0 - 0.6) +
                 image[0, 1] * (0.3 - 0.0) * (1.0 - 0.6) +
                 image[1, 0] * (1.0 - 0.3) * (0.6 - 0.0) +
                 image[1, 1] * (0.3 - 0.0) * (0.6 - 0.0))
    assert_almost_equal(intensities[0], expected0)

    #                  y  x
    expected1 = (image[0, 0] * (1.0 - 0.1) * (1.0 - 0.8) +
                 image[0, 1] * (0.1 - 0.0) * (1.0 - 0.8) +
                 image[1, 0] * (1.0 - 0.1) * (0.8 - 0.0) +
                 image[1, 1] * (0.1 - 0.0) * (0.8 - 0.0))
    assert_almost_equal(intensities[1], expected1)

    #                  y  x
    expected2 = (image[2, 0] * (1.0 - 0.4) * (3.0 - 2.2) +
                 image[2, 1] * (0.4 - 0.0) * (3.0 - 2.2) +
                 image[3, 0] * (1.0 - 0.4) * (2.2 - 2.0) +
                 image[3, 1] * (0.4 - 0.0) * (2.2 - 2.0))

    assert_almost_equal(intensities[2], expected2)

    assert_almost_equal(intensities[3], image[2, 1])

    intensity = interpolation(image, coordinates[2])
    assert_almost_equal(intensity, expected2)
