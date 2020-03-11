import numpy as np
from numpy.testing import assert_array_almost_equal

from scipy.spatial.transform import Rotation

from tadataka.camera import CameraModel, CameraParameters
from tadataka.warp import warp, Warp2D, Warp3D
from tadataka.pose import WorldPose


def test_warp2d():
    # rotate (3 / 4) * pi around the y-axis
    rotation0 = Rotation.from_rotvec([0, (3 / 4) * np.pi, 0])
    t0 = np.array([0, 0, 3])
    pose0 = WorldPose(rotation0, t0)

    # rotate - pi / 2 around the y-axis
    rotation1 = Rotation.from_rotvec([0, -np.pi / 2, 0])
    t1 = np.array([4, 0, 3])
    pose1 = WorldPose(rotation1, t1)

    warp3d = Warp3D(pose0, pose1)

    P0 = np.array([
        [0, 0, 2 * np.sqrt(2)],
        [0, 0, 4 * np.sqrt(2)]
    ])

    # world point should be at
    # [[2, 0, 1], [4, 0, 7]]
    # [[2, 0, -2], [0, 0, 4]]
    assert_array_almost_equal(warp3d(P0), [[-2, 0, 2], [-4, 0, 0]])

    # accept 1d input
    # center of camera 0 seen from camera 1
    assert_array_almost_equal(warp3d(np.zeros(3)), [0, 0, 4])


def test_warp3d():
    rotation = Rotation.from_rotvec([0, np.pi/2, 0])

    t0 = np.array([0, 0, 3])
    pose0 = WorldPose(rotation, t0)

    t1 = np.array([0, 0, 4])
    pose1 = WorldPose(rotation, t1)

    warp3d = Warp3D(pose0, pose1)

    xs0 = np.array([
        [0, 0],
        [0, -1]
    ])
    depths0 = np.array([2, 4])

    xs1, depths1 = warp(warp3d, xs0, depths0)

    assert_array_almost_equal(xs1, [[0.5, 0], [0.25, -1]])
    assert_array_almost_equal(depths1, [2, 4])

    camera_model0 = CameraModel(
        CameraParameters(focal_length=[2, 2], offset=[0, 0]),
        distortion_model=None
    )
    camera_model1 = CameraModel(
        CameraParameters(focal_length=[3, 3], offset=[0, 0]),
        distortion_model=None
    )

    us0 = 2.0 * xs0
    warp2d = Warp2D(camera_model0, camera_model1, warp3d)
    us1, depths1 = warp2d(us0, depths0)
    assert_array_almost_equal(us1, 3.0 * xs1)
