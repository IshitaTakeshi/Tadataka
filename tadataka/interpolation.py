from numba import njit
import numpy as np

from tadataka.decorator import allow_1d
from tadataka.utils import is_in_image_range


@njit
def interpolation__(image, c, af, bf, ai, bi):
    cx, cy = c

    ax, ay = af
    axi, ayi = ai

    if ax == cx and ay == cy:
        return image[ayi, axi]

    bx, by = bf
    bxi, byi = bi

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
def interpolation1d_(image, c):
    af = np.floor(c)
    bf = af + 1.0
    ai = af.astype(np.int64)
    bi = bf.astype(np.int64)

    return interpolation__(image, c, af, bf, ai, bi)


@njit
def interpolation2d_(image, C):
    AF = np.floor(C)
    BF = AF + 1.0
    AI = AF.astype(np.int64)
    BI = BF.astype(np.int64)

    N = C.shape[0]
    intensities = np.empty(N)
    for i in range(N):
        intensities[i] = interpolation__(image, C[i],
                                         AF[i], BF[i], AI[i], BI[i])
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
