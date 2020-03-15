from numba import njit
import numpy as np

from tadataka.decorator import allow_1d
from tadataka.utils import is_in_image_range


@njit
def interpolation__(image, c):
    cx, cy = c

    l = np.floor(c)
    lx, ly = l
    lxi, lyi = l.astype(np.int64)

    if lx == cx and ly == cy:
        return image[lyi, lxi]

    u = l + 1.0
    ux, uy = u
    uxi, uyi = u.astype(np.int64)

    if lx == cx:
        return (image[lyi, lxi] * (ux - cx) * (uy - cy) +
                image[uyi, lxi] * (ux - cx) * (cy - ly))

    if ly == cy:
        return (image[lyi, lxi] * (ux - cx) * (uy - cy) +
                image[lyi, uxi] * (cx - lx) * (uy - cy))

    return (image[lyi, lxi] * (ux - cx) * (uy - cy) +
            image[lyi, uxi] * (cx - lx) * (uy - cy) +
            image[uyi, lxi] * (ux - cx) * (cy - ly) +
            image[uyi, uxi] * (cx - lx) * (cy - ly))


@njit
def interpolation_(image, C):
    U = np.empty(C.shape)
    for i in range(C.shape[0]):
        U[i] = interpolation__(image, C[i])
    return U


@allow_1d(which_argument=1)
def interpolation(image, C):
    """
    Args:
        image (np.ndarary): gray scale image
        coordinates (np.ndarray): coordinates of shape (n_coordinates, 2)
    """

    if not np.ndim(image) == 2:
        raise ValueError("Image have to be a two dimensional array")

    mask = is_in_image_range(C, image.shape)
    if not mask.all():
        raise ValueError(
            "Coordinates {} out of image range".format(C[~mask])
        )

    return interpolation_(image, C)
