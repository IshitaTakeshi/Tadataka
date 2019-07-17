from autograd import numpy as np


def transform_each(rotations, translations, points):
    # same as computing
    # for R, t in zip(rotations, translations):
    #     for p in points:
    #         np.dot(R, p) + t

    # translations.shape == (n_viewpoints, 3)
    # rotations.shape == (n_viewpoints, 3)
    # points.shape == (n_points, 3)

    assert(rotations.shape[0] == translations.shape[0])
    assert(rotations.shape[1:3] == (3, 3))
    assert(translations.shape[1] == 3)

    # reshape translations to align the shape to the rotated points
    # translations.shapee == (n_viewpoints, 1, 3)
    n_viewpoints = translations.shape[0]
    translations = translations.reshape(n_viewpoints, 1, 3)

    # points.shape = (n_viewpoints, n_points, 3)
    # l : n_points
    # i : n_viewpoints
    points = np.einsum('ijk,lk->ilj', rotations, points)
    points = points + translations
    return points


def inv_transform_each(rotations, translations, points):
    # same as computing
    # for R, t in zip(rotations, translations):
    #     for p in points:
    #         np.dot(R.T, p-t)

    # we separate the computation of np.dot(R.T, p-t) to
    # np.dot(R.T, p) - np.dot(R.T, t)

    rotations = np.swapaxes(rotations, 1, 2)  # [R.T for R in rotations]
    # points.shape = (n_viewpoints, n_points, 3)
    points = np.einsum('ijk,lk->ilj', rotations, points)
    # translations.shape = (n_viewpoints, 3)
    translations = np.einsum('ijk,ik->ij', rotations, translations)
    shape = translations.shape
    translations = translations.reshape(shape[0], 1, shape[1])
    return points - translations


def transform(R, t, X):
    return np.dot(R, X.T).T + t
