import numpy as np
from numpy.testing import assert_array_almost_equal

from scipy.spatial.transform import Rotation
from tadataka.pose import WorldPose
from tadataka.warp import Warp2D
from tadataka.camera import CameraModel, CameraParameters
from tadataka.vo.semi_dense.propagation import (
    propagate_variance, coordinates
)


def test_propagate_variance():
    inv_depths0 = np.array([4.0, 1.0, 2.0])
    inv_depths1 = np.array([2.0, 3.0, 1.0])
    variances0 = np.array([0.5, 1.0, 4.0])

    assert_array_almost_equal(
        propagate_variance(inv_depths0, inv_depths1, variances0, 1.0),
        np.power(inv_depths1 / inv_depths0, 4) * variances0 + 1.0
    )


def test_coordinates():
    depth_map0 = np.array([
        [2.0, 3.0],
        [8.0, 2.0]
    ])
    cm = CameraModel(
        CameraParameters(focal_length=[1, 1], offset=[0.5, 0.5]),
        distortion_model=None
    )

    rotation = Rotation.identity()
    pose0 = WorldPose(rotation, np.array([0, 0, 0]))
    pose1 = WorldPose(rotation, np.array([-2, 0, -2]))
    warp10 = Warp2D(cm, cm, pose0, pose1)
    # xs             #  x  y   z
    # [-0.5, -0.5],  # [0, 0]  2.0
    # [0.5, -0.5],   # [1, 0]  3.0
    # [-0.5, 0.5],   # [0, 1]  8.0
    # [0.5, 0.5]     # [1, 1]  2.0

    # inv projection
    # [-1.0, -1.0, 2.0],
    # [1.5, -1.5, 3.0],
    # [-4.0, 4.0, 8.0],
    # [1.0, 1.0, 2.0],

    # after transform
    # [1.0, -1.0, 4.0],
    # [3.5, -1.5, 5.0],
    # [-2.0, 4.0, 10.0],
    # [3.0, 1.0, 4.0]

    # reprojection
    # [0.25, -0.25]
    # [0.7, -0.3]
    # [-0.2, 0.4]
    # [0.75, 0.25]

    # image coordinates
    # [0.75, 0.25]
    # [1.2, 0.2]
    # [0.3, 0.9]
    # [1.25, 0.75]

    us0, us1, depths0, depths1 = coordinates(warp10, depth_map0)
    # image coordinates
    assert_array_almost_equal(us1,
        [[0.75, 0.25],
         [1.2, 0.2],
         [0.3, 0.9],
         [1.25, 0.75]]
    )
