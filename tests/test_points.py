from numpy.testing import assert_array_equal
from autograd import numpy as np

from vitamine.points import get_point_indices, init_empty_points, concat_points


def test_init_empty_points():
    points = init_empty_points()
    assert(points.shape == (0, 3))


def test_get_point_indices():
    points = init_empty_points()
    new_points = np.arange(30).reshape(10, 3)
    point_indices = get_point_indices(points, new_points)
    assert_array_equal(point_indices, np.arange(0, 10))

    points = np.arange(30, 60).reshape(10, 3)
    new_points = np.arange(30, 60).reshape(10, 3)
    point_indices = get_point_indices(points, new_points)
    assert_array_equal(point_indices, np.arange(10, 20))


def test_concat_points():
    points = np.arange(0, 60).reshape(20, 3)
    new_points = np.arange(60, 90).reshape(10, 3)
    points, point_indices = concat_points(points, new_points)
    assert_array_equal(points, np.arange(90).reshape(30, 3))
    assert_array_equal(point_indices, np.arange(20, 30))
