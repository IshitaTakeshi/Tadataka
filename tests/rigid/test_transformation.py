from autograd import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from vitamine.rigid.transformation import (
    inv_transform_all, transform_all, world_to_camera)


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


def test_poses_from_world():
    # we describe rotations below according to the right hand rule
    # along with the camera_locations in the world coordinate system

    rotations = np.array([
        # rotate camera 90 degrees along the axis [0, 1, 0]
        [[0, 0, 1],
         [0, 1, 0],
         [-1, 0, 0]],
        # rotate camera 45 degrees along the axis [1, 0, 0]
        [[1, 0, 0],
         [0, 1, -1],
         [0, 1, 1]]
    ])

    camera_locations = np.array([
        [1, 0, 0],  # move 1.0 to the right
        [0, 0, -1]  # move 1.0 to the back
    ])

    points = np.array([
        [0, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [1, 0, -0.5]
    ])

    # relative point coordinates from each camera
    expected = np.array([
        [[0, 0, -1],
         [0, 0, -2],
         [-1, 0, -1],
         [0.5, 0, 0]],
        [[0, 1, 1],
         [-1, 1, 1],
         [0, 2, 2],
         [1, 0.5, 0.5]]
    ])

    assert_array_almost_equal(
        inv_transform_all(rotations, camera_locations, points),
        expected
    )

    rotations, translations = world_to_camera(rotations, camera_locations)
    assert_array_almost_equal(
        transform_all(rotations, translations, points),
        expected
    )
