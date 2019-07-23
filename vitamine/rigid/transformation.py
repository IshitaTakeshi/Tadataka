from autograd import numpy as np


def transform_all(rotations, translations, points):
    # same as computing
    # for R, t in zip(rotations, translations):
    #     for p in points:
    #         np.dot(R, p) + t

    # translations.shape == (n_viewpoints, 3)
    # rotations.shape == (n_viewpoints, 3)
    # points.shape == (n_points, 3)

    assert(rotations.shape[0] == translations.shape[0])
    assert(rotations.shape[1:3] == (3, 3))
    assert(translations.shape[1] == 3)

    # reshape translations to align the shape to the rotated points
    # translations.shapee == (n_viewpoints, 1, 3)
    n_viewpoints = translations.shape[0]
    translations = translations.reshape(n_viewpoints, 1, 3)

    # points.shape = (n_viewpoints, n_points, 3)
    # l : n_points
    # i : n_viewpoints
    points = np.einsum('ijk,lk->ilj', rotations, points)
    points = points + translations
    return points


def inv_transform_all(rotations, translations, points):
    assert(rotations.shape[0] == translations.shape[0])
    assert(rotations.shape[1:3] == (3, 3))
    assert(translations.shape[1] == 3)

    # same as computing
    # for R, t in zip(rotations, translations):
    #     for p in points:
    #         np.dot(R.T, p-t)

    # we separate the computation of np.dot(R.T, p-t) to
    # np.dot(R.T, p) - np.dot(R.T, t)

    rotations = np.swapaxes(rotations, 1, 2)  # [R.T for R in rotations]

    # points.shape = (n_viewpoints, n_points, 3)
    points = np.einsum('ijk,lk->ilj', rotations, points)

    # translations.shape = (n_viewpoints, 3)
    translations = np.einsum('ijk,ik->ij', rotations, translations)
    shape = translations.shape
    translations = translations.reshape(shape[0], 1, shape[1])
    return points - translations


def transpose_each(rotations):
    return np.swapaxes(rotations, 1, 2)


def rotate_each(rotations, points):
    # the computation is same as below
    # [np.dot(R, p) for R, p in zip(rotations, points)
    assert(rotations.shape[0] == points.shape[0])
    assert(rotations.shape[1:3] == (3, 3))
    assert(points.shape[1] == 3)

    return np.einsum('ijk,ik->ij', rotations, points)


def convert_coordinates(R, t):
    R = transpose_each(R)
    t = -rotate_each(R, t)
    return R, t


def world_to_camera(camera_rotations, camera_locations):
    """
    Given rotations and camera locations in the world coordinate system,
    return rotations and translations for rigid transformation

    camera_rotations:
        Camera rotations in the world coordinate system
    camera_locations:
        Camera locations in the world coordinate system
    """

    rotations, translations =\
        convert_coordinates(camera_rotations, camera_locations)
    return rotations, translations


def camera_to_world(rotations, translations):
    """
    Given rotations and translations in the camera coordinate system,
    return camera rotations and locations

    rotations:
        Relative rotations of the world coordinate system
        seen from cameras
    translations:
        Relative locations of the world origin
    """

    camera_rotations, camera_locations =\
        convert_coordinates(rotations, translations)
    return camera_rotations, camera_locations



def transform(R, t, X):
    return np.dot(R, X.T).T + t


def inv_transform(R, t, P):
    P = P - t
    return np.dot(R.T, P.T).T
