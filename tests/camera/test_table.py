import numpy as np
from numpy.testing import assert_array_almost_equal

from tadataka.camera import CameraModel, CameraParameters
from tadataka.coordinates import image_coordinates
from tadataka.camera.table import NoramlizationMapTable


def test_normalize():
    camera_model = CameraModel(
        CameraParameters(focal_length=[1.0, 0.5], offset=[1.0, 0.5]),
        distortion_model=None
    )

    width, height = 3, 2
    image_shape = (height, width)

    table = NoramlizationMapTable(camera_model, image_shape)

    us_map_0 = np.array([
        [0, 1, 2],
        [0, 1, 2]
    ])

    us_map_1 = np.array([
        [0, 0, 0],
        [1, 1, 1]
    ])

    xs_map_0 = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
    ])

    xs_map_1 = np.array([
        [-1, -1, -1],
        [1, 1, 1],
    ])

    us_query = np.array([
        [1.5, 0.5],
        [0.3, 1.0]
    ])
    xs_pred = table.normalize(us_query)
    us_pred = camera_model.unnormalize(xs_pred)

    assert_array_almost_equal(us_pred, us_query)
