import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from skimage.color import rgb2gray
from skimage.data import astronaut

from tadataka.flow_estimation.extrema_tracker import (
    ExtremaTracker, maximize_one, step, compute_regularizer_map)
from tadataka.flow_estimation.image_curvature import compute_image_curvature


def test_compute_regularizer_map():
    def f(x):
        return np.dot(x, x)

    assert_array_equal(
        compute_regularizer_map(f),
        1 - np.array([
            [2, 1, 2],
            [1, 0, 1],
            [2, 1, 2]
        ])
    )


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


def test_maximize_one():
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

    R = np.zeros((3, 3))

    initial_coordinates = np.array([
        [3, 3],
        [2, 2],
        [3, 2],
        [5, 1]
    ])
    for p0 in initial_coordinates:
        p = maximize_one(curvature, R, p0)
        assert_array_equal(p, [3, 3])

    M = -np.inf
    curvature = np.array([
    # x  0  1  2  3  4  5  6    # y
        [M, M, M, M, M, M, M],  # 0
        [M, 0, 0, 0, 0, 0, M],  # 1
        [M, 0, 2, 2, 2, 0, M],  # 2
        [M, 0, 2, 1, 2, 0, M],  # 3
        [M, 0, 2, 2, 4, 0, M],  # 4
        [M, 0, 0, 0, 0, 0, M],  # 5
        [M, M, M, M, M, M, M]   # 6
    ], dtype=np.float64)

    R = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    p = maximize_one(curvature, R, [3, 2])
    assert_array_equal(p, [4, 4])


def test_extrema_tracker():
    curvature = compute_image_curvature(rgb2gray(astronaut()))

    # coordinates are represented in [xs, ys] format
    initial_coordinates = np.array([
        [10.4, 20.5],
        [30.2, 40.3],
        [60.0, 50.2],
        [40.1, 30.8],
        [80.1, 60.5]
    ])

    R = 1 - np.array([
        [2, 1, 2],
        [1, 0, 1],
        [2, 1, 2]
    ])

    def regularizer(x):
        return 1 - np.dot(x, x)

    def run(lambda_):
        # disable regularization
        extrema_tracker = ExtremaTracker(curvature, lambda_=lambda_,
                                         regularizer=regularizer)
        coordinates = extrema_tracker.optimize(initial_coordinates)

        # the resulting point should have the maxium energy
        # among its neighbors
        # regularizer term can be omitted because lambda_ = 0.0
        for px, py in coordinates.astype(np.int64):
            E = curvature[py, px] + R[1+0, 1+0]

            # E is maximum among its neighbors
            assert(curvature[py-1, px-1] + lambda_ * R[1-1, 1-1] < E)
            assert(curvature[py-1, px-0] + lambda_ * R[1-1, 1-0] < E)
            assert(curvature[py-1, px+1] + lambda_ * R[1-1, 1+1] < E)
            assert(curvature[py+0, px-1] + lambda_ * R[1+0, 1-1] < E)

            assert(curvature[py+0, px+1] + lambda_ * R[1+0, 1+1] < E)
            assert(curvature[py+1, px-1] + lambda_ * R[1+1, 1-1] < E)
            assert(curvature[py+1, px-0] + lambda_ * R[1+1, 1-0] < E)
            assert(curvature[py+1, px+1] + lambda_ * R[1+1, 1+1] < E)

    run(lambda_=0.0)
    run(lambda_=1.0)
    run(lambda_=1e10)
