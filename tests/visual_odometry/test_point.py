from numpy.testing import assert_array_equal
from autograd import numpy as np

from vitamine.visual_odometry.point import Points


def test_points():
    points = Points()

    points_ = np.arange(30).reshape(10, 3)
    point_indices = points.add(points_)
    assert_array_equal(point_indices, np.arange(0, 10))

    points_ = np.arange(30, 60).reshape(10, 3)
    point_indices = points.add(points_)
    assert_array_equal(point_indices, np.arange(10, 20))

    assert_array_equal(points.get(np.arange(5, 15)),
                       np.arange(15, 45).reshape(10, 3))

