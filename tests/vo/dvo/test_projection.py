from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np

from tadataka.camera import CameraParameters
from tadataka.coordinates import compute_pixel_coordinates
from tadataka.projection import inverse_projection, projection


def test_inverse_projection():
    camera_parameters = CameraParameters(focal_length=[2, 2], offset=[1, -2])
    depth_map = np.array(
        [[1, 2, 5],
         [3, 4, 6]]
    )

    S = inverse_projection(camera_parameters, depth_map)

    GT = np.array([
        [(0-1)*1 / 2, (0+2)*1 / 2, 1],  # 3d point corresponding to (x0, y0)
        [(1-1)*2 / 2, (0+2)*2 / 2, 2],  # 3d point corresponding to (x0, y0)
        [(2-1)*5 / 2, (0+2)*5 / 2, 5],  # 3d point corresponding to (x1, y0)
        [(0-1)*3 / 2, (1+2)*3 / 2, 3],  # 3d point corresponding to (x1, y1)
        [(1-1)*4 / 2, (1+2)*4 / 2, 4],  # 3d point corresponding to (x2, y1)
        [(2-1)*6 / 2, (1+2)*6 / 2, 6]   # 3d point corresponding to (x2, y1)
    ])

    assert_array_equal(S, GT)

    # is really the inverse
    assert_array_equal(
        compute_pixel_coordinates(depth_map.shape),
        projection(camera_parameters, S)
    )


def test_projection():
    camera_parameters = CameraParameters([12, 16], [3, 4])

    S = np.array([
        [1, 2, 3],
        [4, 5, 2]
    ])

    P = projection(camera_parameters, S)

    GT = np.array([
        [1 * 12 / 3 + 3, 2 * 16 / 3 + 4],
        [4 * 12 / 2 + 3, 5 * 16 / 2 + 4]
    ])

    assert_array_almost_equal(P, GT)

    # In the case Z <= 0, np.nan is returned
    S = np.array([
        [1, 2, 0],
        [4, 5, -1]
    ])
    P = projection(camera_parameters, S)
    GT = np.array([
        [np.nan, np.nan],
        [np.nan, np.nan]
    ])
    assert_array_equal(P, GT)
