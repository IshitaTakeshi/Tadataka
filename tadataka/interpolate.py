import numpy as np
from tadataka.utils import is_in_image_range


def interpolate(image, coordinates):
    assert(is_in_image_range(coordinates, image.shape).all())
    P = coordinates.astype(np.int64)
    D = coordinates - P
    dx, dy = D[:, 0], D[:, 1]
    ixs, iys = P[:, 0], P[:, 1]

    return (dx * dy * image[iys+1, ixs+1] +
            dy * (1 - dx) * image[iys+1, ixs] +
            dx * (1 - dy) * image[iys, ixs+1] +
            (1 - dx) * (1 - dy) * image[iys, ixs])
