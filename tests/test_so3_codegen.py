from numpy.testing import assert_array_almost_equal
import numpy as np

from tadataka.so3_codegen import projection, pose_jacobian, point_jacobian


def test_projection():
    points = np.array([
        [4, -2, -3],
        [-3, 2, 1],
        [5, 3, -6]
    ])

    omegas = np.array([
        [0, 0, 0],          # R : (x, y, z) -> (x, y, z)
        [np.pi / 2, 0, 0],  # R : (x, y, z) -> (x, -z, y)
        [0, 0, np.pi]       # R : (x, y, z) -> (-x, -y, z)
    ])

    translations = np.array([
        [4, -8, 1],
        [3, -1, 2],
        [2, 0, -4]
    ])

    expected = np.array([
        [-4, 5],
        [0, -0.5],
        [0.3, 0.3]
    ])

    for i, (omega, t, point) in enumerate(zip(omegas, translations, points)):
        pose = np.concatenate((omega, t))

        assert_array_almost_equal(
            projection(pose, point),
            expected[i]
        )
