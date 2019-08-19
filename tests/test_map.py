from autograd import numpy as np
from numpy.testing import assert_array_equal

from vitamine.map import Map


def test_map():
    map_ = Map()

    assert(not map_.is_initialized)

    camera_omegas = np.array([
        [0, 0, 1],
        [1, 0, 0]
    ])

    camera_locations = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    points = np.array([
        [0, -1, -2],
        [-3, -4, -5]
    ])

    map_.add(camera_omegas, camera_locations, points)

    # when initialization
    camera_omegas_, camera_locations_, points_ = map_.get()
    assert_array_equal(camera_omegas_, camera_omegas)
    assert_array_equal(camera_locations_, camera_locations)
    assert_array_equal(points_, points)

    assert(map_.is_initialized)

    camera_omegas = np.array([
        [0, 0, -1],
        [0, -1, 0],
    ])

    camera_locations = np.array([
        [4, 5, 1],
        [2, 3, 1],
    ])

    points = np.array([
        [0, 1, 2],
        [-2, -3, -1]
    ])

    map_.add(camera_omegas, camera_locations, points)

    camera_omegas_, camera_locations_, points_ = map_.get()

    assert_array_equal(
        camera_omegas_,
        np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, -1],
            [0, -1, 0]
        ])
    )

    assert_array_equal(camera_locations_,
        np.array([
            [1, 2, 3],
            [4, 5, 6],
            [4, 5, 1],
            [2, 3, 1]
        ])
    )

    assert_array_equal(
        points_,
        np.array([
            [0, -1, -2],
            [-3, -4, -5],
            [0, 1, 2],
            [-2, -3, -1]
        ])
    )
