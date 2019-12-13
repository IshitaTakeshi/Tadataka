import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from skimage.color import rgb2gray
from skimage.data import astronaut

from tadataka.flow_estimation.extrema_tracker import (
    ExtremaTracker, get_neighbors, energy, regularize, maximize_)
from tadataka.flow_estimation.image_curvature import compute_image_curvature


def test_get_neighbors():
    # 'get_neighbors' reuqires image shape in the (height, width) order,
    # whereras the points and their neighbors below are in (x, y)
    width, height = 5, 4
    image_shape = (height, width)

    # the case all neighbors fit in the image range
    expected = np.array([
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [0, 3], [1, 3], [2, 3],
    ])
    assert_array_equal(get_neighbors(np.array([1, 2]), image_shape), expected)

    # [x, 4] are removed because they are out of the image range
    expected = np.array([
        [0, 2], [1, 2], [2, 2],
        [0, 3], [1, 3], [2, 3],
        # [0, 4], [1, 4], [2, 4]
    ])
    assert_array_equal(get_neighbors(np.array([1, 3]), image_shape), expected)

    # [5, y] are removed because they are out of the image range
    expected = np.array([
        [3, 1], [4, 1], # [5, 1],
        [3, 2], [4, 2], # [5, 2],
        [3, 3], [4, 3], # [5, 3]
    ])
    assert_array_equal(get_neighbors(np.array([4, 2]), image_shape), expected)


def test_regularizer():
    p0 = np.array([40, 50])
    image_shape = (100, 100)
    neighbors = get_neighbors(p0, image_shape)

    squared_norms = np.array([
        2, 1, 2,
        1, 0, 1,
        2, 1, 2
    ])

    assert(len(squared_norms) == len(neighbors))
    for squared_norm, neighbor in zip(squared_norms, neighbors):
        assert_array_almost_equal(regularize(neighbor, p0), 1 - squared_norm)


def test_energy():
    curvature = np.array([[4, -2, -7],
                          [-1, 3, 9],
                          [8, 5, -1]])

    R = np.array([[-1, 0, -1],
                  [0, 1, 0],
                  [-1, 0, -1]])

    lambda_ = 2
    image_shape = (100, 100)

    p0 = np.array([1, 1])

    # compute errors at p0 and its 8 neighbors
    neighbors = get_neighbors(p0, image_shape)
    energy_pred = energy(curvature, p0, neighbors, lambda_)

    energy_true = (curvature + lambda_ * R).flatten()
    assert_array_almost_equal(energy_pred, energy_true)


def test_maximize_():
    curvature = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 2, 2, 0],
        [0, 2, 4, 2, 0],
        [0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0]
    ])

    p0 = np.array([2, 2])
    p = maximize_(curvature, p0, lambda_=0.0, max_iter=20)
    assert_array_equal(p0, p)

    # p0 = np.array([2, 2])
    # p = maximize_(curvature, p0, lambda_=0.0, max_iter=20)
    # assert_array_equal(p0, p)


def test_extrema_tracker():
    # diff to 8 neighbors
    diffs = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 1],
        [1, -1], [1, 0], [1, 1]
    ])

    curvature = compute_image_curvature(rgb2gray(astronaut()))

    # coordinates are represented in [xs, ys] format
    initial_coordinates = np.array([
        [10, 20],
        [30, 40],
        [60, 50],
        [40, 30]
    ])

    lambda_ = 0.0

    # disable regularization
    extrema_tracker = ExtremaTracker(curvature, lambda_)
    coordinates = extrema_tracker.optimize(initial_coordinates)

    # the resulting point should have the maxium energy
    # among its neighbors
    for p in coordinates:
        neighbors = get_neighbors(p, curvature.shape)
        E0 = energy(curvature, p, np.atleast_2d(p), lambda_)[0]
        E = energy(curvature, p, neighbors, lambda_)
        assert((E0 >= E).all())

    # apply very strong regularization
    extrema_tracker = ExtremaTracker(curvature, lambda_=1e2)
    coordinates = extrema_tracker.optimize(initial_coordinates)
    # the points should not move from the initial values
    assert_array_equal(coordinates, initial_coordinates)
