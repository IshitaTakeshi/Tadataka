import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from scipy.spatial.transform import Rotation

from tadataka.camera import CameraModel, CameraParameters
from tadataka.projection import pi, inv_pi


def test_pi():
    P = np.array([
        [0, 0, 0],
        [1, 4, 2],
        [-1, 3, 5],
    ], dtype=np.float64)

    assert_array_almost_equal(
        pi(P),
        [[0., 0.], [0.5, 2.0], [-0.2, 0.6]]
    )

    assert_array_almost_equal(pi(np.array([0., 0., 0.])), [0, 0])
    assert_array_almost_equal(pi(np.array([3., 5., 5.])), [0.6, 1.0])


def test_inv_pi():
    xs = np.array([
        [0.5, 2.0],
        [-0.2, 0.6]
    ])
    depths = np.array([2.0, 5.0])

    assert_array_almost_equal(
        inv_pi(xs, depths),
        [[1.0, 4.0, 2.0],
         [-1.0, 3.0, 5.0]]
    )

    x = np.array([0.5, 2.0])
    depth = 2.0
    assert_array_almost_equal(inv_pi(x, depth), [1.0, 4.0, 2.0])
