import numpy as np

from tadataka.decorator import allow_1d
from tadataka.utils import is_in_image_range
from rust_bindings import interpolation as _interpolation


@allow_1d(which_argument=1)
def interpolation_(image, C):
    return _interpolation.interpolation(image, C)


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
