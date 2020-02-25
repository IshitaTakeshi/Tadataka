import numpy as np

from tadataka.decorator import allow_1d


@allow_1d(which_argument=1)
def interpolation(image, C):
    """
    Args:
        image (np.ndarary): gray scale image
        coordinates (np.ndarray): coordinates of shape (n_coordinates, 2)
    """
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
