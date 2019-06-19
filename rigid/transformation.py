from autograd import numpy as np


def transform_each(rotations, translations, points):
    # translations.shape == (n_viewpoints, 3)
    # rotations.shape == (n_viewpoints, 3)
    # points.shape == (n_points, 3)

    assert(rotations.shape[0] == translations.shape[0])
    assert(rotations.shape[1:3] == (3, 3))
    assert(translations.shape[1] == 3)

    # reshape translations to align the shape to the rotated points
    # translations.shapee == (1, n_viewpoints, 3)
    translations = translations[np.newaxis]

    # points.shape = (n_points, n_viewpoints, 3)
    # i : n_viewpoints
    # l : n_points
    points = np.einsum('ijk,lk->lij', rotations, points)
    points = points + translations
    return points
