import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from scipy.spatial.transform import Rotation

from tadataka.camera import CameraModel, CameraParameters
from tadataka.pose import LocalPose, WorldPose
from tadataka.projection import pi, warp, Warp


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


def test_warp():
    coordinates = np.array([
        [0, 1],
        [2, 5],
        [-4, -1],
        [3, -2]
    ])
    depths = np.array([4, 5, 2, 3])

    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    t = np.array([1, -1, 5])
    # [[0, 4, 4],
    #  [10, 25, 5],
    #  [-8, -2, 2],
    #  [9, -6, 3]]
    # [[1, -5, 9],
    #  [11, -6, 30],
    #  [-7, -3, 3]]
    #  [10, -4, -1]
    assert_array_almost_equal(
        warp(coordinates, depths, R, t),
        [[1 / 9, -5 / 9],
         [11 / 30, -6 / 30],
         [-7 / 3, -3 / 3],
         [10 / -1, -4 / -1]]
    )


def test_warp_class():
    cm0 = CameraModel(
        CameraParameters(focal_length=[1, 1], offset=[0, 0]),
        distortion_model=None
    )
    cm1 = CameraModel(
        CameraParameters(focal_length=[1, 1], offset=[0, 0]),
        distortion_model=None
    )

    # rotate pi / 4 around the y-axis
    rotation0 = Rotation.from_rotvec([0, np.pi/4, 0])
    t0 = np.array([0, 0, 3])
    pose0 = WorldPose(rotation0, t0)

    u0 = np.array([0, 0])
    depth0 = 2 * np.sqrt(2)

    # world point should be at [2, 0, 5]

    # rotate - pi / 2 around the y-axis
    rotation1 = Rotation.from_rotvec([0, -np.pi/2, 0])
    t1 = np.array([4, 0, 3])
    pose1 = WorldPose(rotation1, t1)

    warp01 = Warp(cm0, cm1, pose0, pose1)
    assert_array_almost_equal(warp01(u0, depth0), [1, 0])
