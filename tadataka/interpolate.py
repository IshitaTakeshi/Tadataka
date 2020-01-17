import numpy as np
from tadataka.utils import is_in_image_range


def interpolate(image, coordinates):
    assert(is_in_image_range(coordinates, image.shape).all())
    P = coordinates.astype(np.int64)
    D = coordinates - P
    dx, dy = D[:, [0]], D[:, [1]]
    ixs, iys = P[:, 0], P[:, 1]

    br = image[iys+1, ixs+1]
    bl = image[iys+1, ixs]
    tr = image[iys, ixs+1]
    tl = image[iys, ixs]

    return (dx * dy * br +
            dy * (1 - dx) * bl +
            dx * (1 - dy) * tr +
            (1 - dx) * (1 - dy) * tl)
