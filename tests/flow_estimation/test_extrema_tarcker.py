from autograd import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from optimization.robustifiers import SquaredRobustifier
from flow_estimation.extrema_tracker import (
        ExtremaTracker, Energy, Neighbors, Regularizer, Maximizer)


def test_neighbors():
    # 'Neighbors' reuqires an image shape which is in
    # the (height, width) order, whereras the points and
    # their neighbors below are in (x, y)
    width, height = 4, 3
    neighbors = Neighbors((height, width))

    # the case all neighbors fit in the image range
    expected = np.array([
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [0, 3], [1, 3], [2, 3],
    ])
    assert_array_equal(neighbors.get(np.array([1, 2])), expected)

    # [x, 4] are removed because they are out of the image range
    expected = np.array([
        [0, 2], [1, 2], [2, 2],
        [0, 3], [1, 3], [2, 3],
        # [0, 4], [1, 4], [2, 4]
    ])
    assert_array_equal(neighbors.get(np.array([1, 3])), expected)


def test_regularizer():
    p0 = np.array([40, 50])
    neighbors = Neighbors((100, 100)).get(p0)
    regularizer = Regularizer(p0, robustifier=SquaredRobustifier())

    squared_norms = np.array([
        2, 1, 2,
        1, 0, 1,
        2, 1, 2
    ])
    assert_array_almost_equal(regularizer.regularize(neighbors),
                              1 - squared_norms)


def test_error():
    K = np.array([[4, -2, -7],
                  [-1, 3, 9],
                  [8, 5, -1]])

    R = np.array([[-1, 0, -1],
                  [0, 1, 0],
                  [-1, 0, -1]])

    lambda_ = 2

    p0 = np.array([1, 1])
    regularizer = Regularizer(p0, robustifier=SquaredRobustifier())
    energy = Energy(K, regularizer, lambda_)

    # compute errors at p0 and its 8 neighbors
    neighbors = Neighbors((100, 100)).get(p0)
    E = K + lambda_ * R
    assert_array_almost_equal(energy.compute(neighbors), E.flatten())


def test_maximizer():
    # always move to right
    class Energy1(object):
        def compute(self, coordinates):
            energies = np.zeros(coordinates.shape[0])
            energies[5] = 1
            return energies

    # move 1 pixel to right 10 times
    maximizer = Maximizer(Energy1(), (100, 100), max_iter=10)
    assert_array_equal(maximizer.search(np.array([4, 3])),
                       np.array([14, 3]))

    # stay at the same pixel
    class Energy2(object):
        def compute(self, coordinates):
            energies = np.zeros(coordinates.shape[0])
            energies[4] = 1
            return energies

    maximizer = Maximizer(Energy2(), (100, 100), max_iter=10)
    assert_array_equal(maximizer.search(np.array([4, 3])),
                       np.array([4, 3]))
