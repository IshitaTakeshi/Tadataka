import numpy as np
from scipy.ndimage import map_coordinates


def interpolation_(I, Q, order):
    # Because 'map_coordinates' requires indices of
    # [row, column] order, the 2nd axis have to be reversed
    # so that it becomes [y, x]

    Q = Q[:, [1, 0]].T

    # Here,
    # Q = [
    #     [y0 y0... y0 ... yi yi... yi ... ym ym ... ym]
    #     [x0 x1... xn ... x0 x1... xn ... x0 x1 ... xn]
    # ]

    # sample pixel sequences from the given image I1
    # warped_image = [
    #     I[y0, x0]  I[y0, x1]  ...  I[y0, xn]
    #     I[y1, x0]  I[y1, x1]  ...  I[y1, xn]
    #     ...
    #     I[ym, x0]  I[ym, x1]  ...  I[ym, xn]
    # ]

    return map_coordinates(I.astype(np.float64), Q,
                           mode="constant", cval=np.nan, order=order)


def interpolation(image, coordinates, order=3):
    """
    Args:
        image (np.ndarary): gray scale image
        coordinates (np.ndarray): coordinates of shape (n_coordinates, 2)
    """

    ndim = np.ndim(coordinates)

    if ndim == 1:
        return interpolation_(image, np.atleast_2d(coordinates), order)[0]

    if ndim == 2:
        return interpolation_(image, coordinates, order)

    raise ValueError("Coordinates have to be 1d or 2d array")
