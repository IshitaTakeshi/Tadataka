from numba import njit
import numpy as np

from tadataka.decorator import allow_1d
from tadataka.utils import is_in_image_range


@njit
def interpolation__(image, c, lf, uf, li, ui):
    cx, cy = c

    lx, ly = lf
    lxi, lyi = li

    if lx == cx and ly == cy:
        return image[lyi, lxi]

    ux, uy = uf
    uxi, uyi = ui

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
def interpolation1d_(image, c):
    lf = np.floor(c)
    uf = lf + 1.0
    li = lf.astype(np.int64)
    ui = uf.astype(np.int64)

    return interpolation__(image, c, lf, uf, li, ui)


@njit
def interpolation2d_(image, C):
    LF = np.floor(C)
    UF = LF + 1.0
    LI = LF.astype(np.int64)
    UI = UF.astype(np.int64)

    N = C.shape[0]
    intensities = np.empty(N)
    for i in range(N):
        intensities[i] = interpolation__(image, C[i],
                                         LF[i], UF[i], LI[i], UI[i])
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
