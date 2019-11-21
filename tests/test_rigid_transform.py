import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from tadataka.rigid_transform import (inv_transform_all, transform_all,
                                           transform_each)
from tadataka.coordinates import world_to_camera, camera_to_world


def test_transform_each():
    points = np.array([
        [1, 2, 5],
        [4, -2, 3],
    ])

    rotations = np.array([
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]],
        [[0, 0, -1],
         [0, 1, 0],
         [1, 0, 0]]
    ])

    translations = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    expected = np.array([
        [2, -3, 5],   # [ 1, -5,  2] + [ 1,  2,  3]
        [1, 3, 10]    # [ -3, -2, 4] + [ 4,  5,  6]
    ])

    assert_array_equal(
        transform_each(rotations, translations, points),
        expected
    )


def test_transform_all():
    points = np.array([
        [1, 2, 5],
        [4, -2, 3],
        [0, 0, 6]
    ])

    rotations = np.array([
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]],
        [[0, 0, -1],
         [0, 1, 0],
         [1, 0, 0]]
    ])

    translations = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    expected = np.array([
        [[2, -3, 5],   # [ 1, -5,  2] + [ 1,  2,  3]
         [5, -1, 1],   # [ 4, -3, -2] + [ 1,  2,  3]
         [1, -4, 3]],  # [ 0, -6,  0] + [ 1,  2,  3]
        [[-1, 7, 7],   # [-5,  2,  1] + [ 4,  5,  6]
         [1, 3, 10],   # [-3, -2,  4] + [ 4,  5,  6]
         [-2, 5, 6]]   # [-6,  0,  0] + [ 4,  5,  6]
    ])

    assert_array_equal(transform_all(rotations, translations, points),
                       expected)


def test_inv_transform_all():
    points = np.array([
        [1, 2, 5],
        [4, -2, 3],
        [0, 0, 6]
    ])

    rotations = np.array([
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]],
        [[0, 0, -1],
         [0, 1, 0],
         [1, 0, 0]]
    ])

    # [R.T for R in rotations]
    # [[1, 0, 0],
    #  [0, 0, 1],
    #  [0, -1, 0]],
    # [[0, 0, 1],
    #  [0, 1, 0],
    #  [-1, 0, 0]]

    translations = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    # p - t
    # [[0, 0, 2],
    #  [3, -4, 0],
    #  [-1, -2, 3]],
    # [[-3, -3, -1],
    #  [0, -7, -3],
    #  [-4, -5, 0]]

    # np.dot(R.T, p-t)
    expected = np.array([
        [[0, 2, 0],
         [3, 0, 4],
         [-1, 3, 2]],
        [[-1, -3, 3],
         [-3, -7, 0],
         [0, -5, 4]]
    ])

    assert_array_equal(inv_transform_all(rotations, translations, points),
                       expected)


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

    rotations, translations = world_to_camera(camera_rotations, camera_locations)

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
        camera_to_world(rotations, translations)

    assert_array_almost_equal(camera_rotations_, camera_rotations)
    assert_array_almost_equal(camera_locations_, camera_locations)
