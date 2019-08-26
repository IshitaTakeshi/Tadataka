from autograd import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from vitamine.dataset.points import donut


def test_donut():
    inner_r = 4
    outer_r = 8
    height = 5
    point_density = 6
    n_viewpoints = 4
    offset = 0.0  # 1e-2

    camera_rotations, camera_locations, points =\
        donut(inner_r, outer_r, height, point_density, n_viewpoints, offset)

    assert_array_equal(points[0*point_density:2*point_density, 1],
                       0 * np.ones(2*point_density))
    assert_array_equal(points[2*point_density:4*point_density, 1],
                       1 * np.ones(2*point_density))
    assert_array_equal(points[4*point_density:6*point_density, 1],
                       2 * np.ones(2*point_density))

    for i in range(0, 2*height, 2):
        assert_array_almost_equal(
            points[(i+0)*point_density:(i+1)*point_density, 0],
            4 * np.array([1, 1/2, -1/2, -1, -1/2, 1/2])
        )
        assert_array_almost_equal(
            points[(i+1)*point_density:(i+2)*point_density, 0],
            8 * np.array([1, 1/2, -1/2, -1, -1/2, 1/2])
        )

    # radius of locations is 6 (= (inner_r + outer_r) / 2)
    assert_array_almost_equal(
        camera_locations,
        np.array([
            [6, 2, 0],
            [0, 2, 6],
            [-6, 2, 0],
            [0, 2, -6]
        ])
    )

    assert_array_almost_equal(
        camera_rotations,
        np.array([
            [0, 0, 0],
            [0, -np.pi / 2, 0],
            [0, -np.pi, 0],
            [0, -3 * np.pi / 2, 0],
        ])
    )
