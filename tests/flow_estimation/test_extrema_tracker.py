import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from skimage.color import rgb2gray
from skimage.data import astronaut

from tadataka.flow_estimation.extrema_tracker import (
    ExtremaTracker, Maximizer, step, compute_regularizer_map)
from tadataka.flow_estimation.image_curvature import compute_image_curvature


def test_step():
    M = np.array([
    # x -1  0  1    # y
        [3, 0, 0],  # -1
        [0, 0, 0],  #  0
        [0, 0, 0],  #  1
    ])
    assert_array_equal(step(M), [-1, -1])

    M = np.array([
    # x -1  0  1    # y
        [0, 0, 0],  # -1
        [0, 0, 4],  #  0
        [0, 0, 0],  #  1
    ])
    assert_array_equal(step(M), [1, 0])

    M = np.array([
    # x -1  0  1    # y
        [0, 8, 7],  # -1
        [1, 3, 0],  #  0
        [4, -1, 0],  #  1
    ])
    assert_array_equal(step(M), [0, -1])


def test_maximizer():
    def regularizer(P):
        y = np.sum(np.power(P, 2), axis=1)
        return y / (y + 10)

    M = -np.inf
    curvature = np.array([
    # x  0  1  2  3  4  5  6    # y
        [M, M, M, M, M, M, M],  # 0
        [M, 0, 0, 0, 0, 0, M],  # 1
        [M, 0, 2, 2, 2, 0, M],  # 2
        [M, 0, 2, 4, 2, 0, M],  # 3
        [M, 0, 2, 2, 2, 0, M],  # 4
        [M, 0, 0, 0, 0, 0, M],  # 5
        [M, M, M, M, M, M, M]   # 6
    ], dtype=np.float64)

    initial_coordinates = np.array([
        [3, 3],
        [2, 2],
        [3, 2],
        [5, 1]
    ])

    maximize = Maximizer(curvature, regularizer, lambda_=0.0)
    for p0 in initial_coordinates:
        p = maximize(p0)
        assert_array_equal(p, [3, 3])

    M = -np.inf
    curvature = np.array([
    # x  0  1  2  3  4  5  6    # y
        [M, M, M, M, M, M, M],  # 0
        [M, 0, 0, 0, 0, 0, M],  # 1
        [M, 0, 2, 0, 2, 0, M],  # 2
        [M, 0, 3, 1, 2, 0, M],  # 3
        [M, 0, 2, 4, 5, 0, M],  # 4
        [M, 0, 0, 0, 0, 0, M],  # 5
        [M, M, M, M, M, M, M]   # 6
    ], dtype=np.float64)

    maximize = Maximizer(curvature, regularizer, lambda_=0.0)
    p = maximize([2, 1])
    assert_array_equal(p, [4, 4])

    maximize = Maximizer(curvature, regularizer, lambda_=30.0)
    p = maximize([2, 1])
    assert_array_equal(p, [2, 1])


def test_compute_regularizer_map():
    def regularizer(P):
        y = np.sum(np.power(P, 2), axis=1)
        return y / (y + 10)

    width, height = 60, 40
    x0, y0 = 10, 20
    R = compute_regularizer_map(regularizer, (height, width), (x0, y0))
    x, y = 40, 10
    assert(R[y, x] == (1 - 1000 / 1010))
    x, y = 20, 15
    assert(R[y, x] == (1 - 125 / 135))


def test_extrema_tracker():
    curvature = compute_image_curvature(rgb2gray(astronaut()))

    # coordinates are represented in [xs, ys] format
    initial_coordinates = np.array([
        [10, 20],
        [30, 40],
        [60, 50],
        [40, 30],
        [80, 60]
    ])

    def regularizer(P):
        y = np.sum(np.power(P, 2), axis=1)
        return y / (y + 10)

    def run(lambda_):
        # disable regularization
        extrema_tracker = ExtremaTracker(curvature, lambda_=lambda_,
                                         regularizer=regularizer)
        coordinates = extrema_tracker.optimize(initial_coordinates)

        # the resulting point should have the maxium energy
        # among its neighbors
        # regularizer term can be omitted because lambda_ = 0.0
        for p0, p in zip(initial_coordinates, coordinates):
            R = compute_regularizer_map(regularizer, curvature.shape, p0)

            x, y = p

            E = curvature[y+0, x+0] + lambda_ * R[y+0, x+0]

            # E is maximum among its neighbors
            assert(curvature[y-1, x-1] + lambda_ * R[y-1, x-1] < E)
            assert(curvature[y-1, x-0] + lambda_ * R[y-1, x-0] < E)
            assert(curvature[y-1, x+1] + lambda_ * R[y-1, x+1] < E)
            assert(curvature[y+0, x-1] + lambda_ * R[y+0, x-1] < E)

            assert(curvature[y+0, x+1] + lambda_ * R[y+0, x+1] < E)
            assert(curvature[y+1, x-1] + lambda_ * R[y+1, x-1] < E)
            assert(curvature[y+1, x-0] + lambda_ * R[y+1, x-0] < E)
            assert(curvature[y+1, x+1] + lambda_ * R[y+1, x+1] < E)

    run(lambda_=0.0)
    run(lambda_=1.0)
    run(lambda_=1e10)

    t = ExtremaTracker(curvature, lambda_=1e8, regularizer=regularizer)
    coordinates = t.optimize(initial_coordinates)
    assert_array_equal(coordinates, initial_coordinates)
