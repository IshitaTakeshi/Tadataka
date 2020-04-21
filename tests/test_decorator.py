import numpy as np
from numpy.testing import assert_array_equal

from tadataka.decorator import allow_1d


def test_allow_1d():
    image = np.random.random((512, 512))

    x, y = 10, 20
    coordinate = np.array([x, y])

    @allow_1d(1)
    def function1(A, C):
        xs, ys = C[:, 0], C[:, 1]
        return A[ys, xs]

    assert(function1(image, coordinate) == image[y, x])

    @allow_1d(0)
    def function2(C, A):
        xs, ys = C[:, 0], C[:, 1]
        return A[ys, xs]

    assert(function2(coordinate, image) == image[y, x])
    coordinates = np.array([
        [10, 3],
        [3, 5],
        [6, 1]
    ])
    assert_array_equal(function2(coordinates, image),
                       image[coordinates[:, 1], coordinates[:, 0]])
