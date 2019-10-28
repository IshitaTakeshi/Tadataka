from autograd import numpy as np


def transform_each(rotations, translations, points):
    assert(rotations.shape[0] == translations.shape[0] == points.shape[0])
    assert(rotations.shape[1:3] == (3, 3))

    # l : n_points
    # i : n_viewpoints
    # points.shape = (N, 3) where N == n_points == n_viewpoints
    points = np.einsum('ijk,ik->ij', rotations, points)
    points = points + translations
    return points



def transform_all(rotations, translations, points):
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


def inv_transform_all(rotations, translations, points):
    assert(rotations.shape[0] == translations.shape[0])
    assert(rotations.shape[1:3] == (3, 3))
    assert(translations.shape[1] == 3)

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


def rotate_each(rotations, points):
    # the computation is same as below
    # [np.dot(R, p) for R, p in zip(rotations, points)
    assert(rotations.shape[0] == points.shape[0])
    assert(rotations.shape[1:3] == (3, 3))
    assert(points.shape[1] == 3)

    return np.einsum('ijk,ik->ij', rotations, points)


def transform(R, t, P):
    """
    R: rotation matrix
    t: translation vector
    P: either a point of shape (3,), or set of points of shape (n_points, 3)
    """
    assert(R.shape == (3, 3))
    assert(t.shape == (3,))
    return np.dot(R, P.T).T + t


def inv_transform(R, t, P):
    P = P - t
    return np.dot(R.T, P.T).T
