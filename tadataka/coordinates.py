import numpy as np

from tadataka.rigid_transform import rotate_each
from tadataka.so3 import rodrigues


def image_coordinates(image_shape):
    height, width = image_shape[0:2]
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    return np.column_stack((xs.flatten(), ys.flatten()))


def transpose_each(rotations):
    return np.swapaxes(rotations, 1, 2)


def convert_translations(rotations, translations):
    return -rotate_each(transpose_each(rotations), translations)


def convert_rotations(rotations):
    return transpose_each(rotations)


def convert_omegas(omegas):
    return -omegas


def convert_coordinates_rotations(rotations, translations):
    return (convert_rotations(rotations),
            convert_translations(rotations, translations))


def convert_coordinates_omegas(omegas, translations):
    return (convert_omegas(omegas),
            convert_translations(rodrigues(omegas), translations))


def convert_coordinates(rotations_or_omegas, translations):
    ndim = np.ndim(rotations_or_omegas)

    if ndim == 3:
        return convert_coordinates_rotations(rotations_or_omegas, translations)

    if ndim == 2:
        return convert_coordinates_omegas(rotations_or_omegas, translations)

    raise ValueError("ndim of 'rotations_or_omegas' must be 2 or 3")


def world_to_local(camera_rotations_or_omegas, camera_locations):
    """
    Given rotations and camera locations in the world coordinate system,
    return rotations and translations for rigid transformation

    camera_rotations_or_omegas:
        Camera rotations in the world coordinate system
    camera_locations:
        Camera locations in the world coordinate system
    """

    return convert_coordinates(camera_rotations_or_omegas, camera_locations)


def local_to_world(rotations_or_omegas, translations):
    """
    Given rotations and translations in the camera coordinate system,
    return camera rotations and locations

    rotations_or_omegas:
        Relative rotations of the world coordinate system
        seen from cameras
    translations:
        Relative locations of the world origin
    """

    return convert_coordinates(rotations_or_omegas, translations)


def yx_to_xy(coordinates):
    return coordinates[:, [1, 0]]


def xy_to_yx(coordinates):
    # this is identical to 'yx_to_xy' but I prefer to name expilictly
    return yx_to_xy(coordinates)
