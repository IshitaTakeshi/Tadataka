import numba
import numpy as np
from scipy.ndimage import map_coordinates


def interpolation_(image, C):
    L = np.floor(C)
    U = L + 1
    LI = L.astype(np.int64)
    UI = U.astype(np.int64)

    CX, CY = C[:, 0], C[:, 1]
    LX, LY = L[:, 0], L[:, 1]
    UX, UY = U[:, 0], U[:, 1]
    LXI, LYI = LI[:, 0], LI[:, 1]
    UXI, UYI = UI[:, 0], UI[:, 1]

    return (image[LYI, LXI] * (UX - CX) * (UY - CY) +
            image[LYI, UXI] * (CX - LX) * (UY - CY) +
            image[UYI, LXI] * (UX - CX) * (CY - LY) +
            image[UYI, UXI] * (CX - LX) * (CY - LY))


def interpolation(image, coordinates):
    """
    Args:
        image (np.ndarary): gray scale image
        coordinates (np.ndarray): coordinates of shape (n_coordinates, 2)
    """

    ndim = np.ndim(coordinates)

    if ndim == 1:
        return interpolation_(image, np.atleast_2d(coordinates))[0]

    if ndim == 2:
        return interpolation_(image, coordinates)

    raise ValueError("Coordinates have to be 1d or 2d array")
