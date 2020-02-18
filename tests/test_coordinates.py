from numpy.testing import assert_array_equal
from tadataka.coordinates import (image_coordinates,
                                  local_to_world, world_to_local)


def test_image_coordinates():
    width, height = 3, 4
    coordinates = image_coordinates(image_shape=[height, width])
    assert_array_equal(
        coordinates,
#         x  y
        [[0, 0],
         [1, 0],
         [2, 0],
         [0, 1],
         [1, 1],
         [2, 1],
         [0, 2],
         [1, 2],
         [2, 2],
         [0, 3],
         [1, 3],
         [2, 3]]
    )


def test_convert_coordinates():
    # we describe rotations below according to the right hand rule
    # along with the camera_locations in the world coordinate system

    camera_rotations = np.array([
        # rotate camera 90 degrees along the axis [0, 1, 0]
        [[0, 0, 1],
         [0, 1, 0],
         [-1, 0, 0]],
        # rotate camera 45 degrees along the axis [1, 0, 0]
        [[1, 0, 0],
         [0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
         [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]]
    ])

    camera_locations = np.array([
        [1, 0, 0],  # move 1.0 to the right
        [0, 0, -1]  # move 1.0 to the back
    ])

    rotations, translations = world_to_local(camera_rotations, camera_locations)

    expected = np.array([
        [0, 0, -1],
        [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]
    ])
    assert_array_equal(translations, expected)

    expected = np.array([
        [[0, 0, -1],
         [0, 1, 0],
         [1, 0, 0]],
        [[1, 0, 0],
         [0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
         [0, -1 / np.sqrt(2), 1 / np.sqrt(2)]]
    ])
    assert_array_equal(rotations, expected)

    camera_rotations_, camera_locations_ =\
        local_to_world(rotations, translations)

    assert_array_almost_equal(camera_rotations_, camera_rotations)
    assert_array_almost_equal(camera_locations_, camera_locations)
