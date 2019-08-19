from autograd import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from skimage.color import rgb2gray
from skimage.data import astronaut

from vitamine.optimization.robustifiers import SquaredRobustifier
from vitamine.flow_estimation.extrema_tracker import (
        ExtremaTracker, Energy, Neighbors, Regularizer, Maximizer)


def test_neighbors():
    # 'Neighbors' reuqires an image shape which is in
    # the (height, width) order, whereras the points and
    # their neighbors below are in (x, y)
    width, height = 5, 4
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

    # [5, y] are removed because they are out of the image range
    expected = np.array([
        [3, 1], [4, 1], # [5, 1],
        [3, 2], [4, 2], # [5, 2],
        [3, 3], [4, 3], # [5, 3]
    ])
    assert_array_equal(neighbors.get(np.array([4, 2])), expected)


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


def test_extrema_tracker():
    # diff to 8 neighbors
    diffs = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 1],
        [1, -1], [1, 0], [1, 1]
    ])

    # this is the image itself but use it as a curvature
    curvature = rgb2gray(astronaut())

    # coordinates are represented in [xs, ys] format
    initial_coordinates = np.array([
        [10, 20],
        [30, 40],
        [60, 50],
        [40, 30]
    ])

    lambda_ = 0.0

    # disable regularization
    extrema_tracker = ExtremaTracker(curvature, initial_coordinates, lambda_)
    coordinates = extrema_tracker.optimize()

    # the resulting points should have the maxium energies
    # compared to their local neighbors
    for p in coordinates:
        energy = Energy(curvature, Regularizer(p), lambda_)
        E0 = energy.compute(p.reshape(1, -1))[0]
        E = energy.compute(p + diffs)
        assert((E0 >= E).all())

    # apply regularization very strongly
    extrema_tracker = ExtremaTracker(curvature, initial_coordinates,
                                     lambda_=1e8)
    coordinates = extrema_tracker.optimize()
    # the points should not move from the initial values
    assert_array_equal(coordinates, initial_coordinates)
