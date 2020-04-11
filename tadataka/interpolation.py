from numba import njit
import numpy as np

from tadataka.decorator import allow_1d
from tadataka.utils import is_in_image_range


@njit
def interpolation1d_(image, c):
    cx, cy = c

    a = np.floor(c)
    ax, ay = a
    axi, ayi = a.astype(np.int64)

    if ax == cx and ay == cy:
        return image[ayi, axi]

    b = a + 1.0
    bx, by = b
    bxi, byi = b.astype(np.int64)

    if ax == cx:
        return (image[ayi, axi] * (bx - cx) * (by - cy) +
                image[byi, axi] * (bx - cx) * (cy - ay))

    if ay == cy:
        return (image[ayi, axi] * (bx - cx) * (by - cy) +
                image[ayi, bxi] * (cx - ax) * (by - cy))

    return (image[ayi, axi] * (bx - cx) * (by - cy) +
            image[ayi, bxi] * (cx - ax) * (by - cy) +
            image[byi, axi] * (bx - cx) * (cy - ay) +
            image[byi, bxi] * (cx - ax) * (cy - ay))


@njit
def interpolation2d_(image, C):
    N = C.shape[0]
    intensities = np.empty(N)
    for i in range(N):
        intensities[i] = interpolation1d_(image, C[i])
    return intensities


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

    return interpolation2d_(image, C)
