import numpy as np

from rigid.rotation import rodrigues
from rigid.transformation import transform_each


def generate_observations(rotations, translations, points, projection):
    assert(points.shape[1] == 3)
    assert(rotations.shape[0] == translations.shape[0])
    assert(translations.shape[1] == 3)
    assert(rotations.shape[1:3] == (3, 3))

    n_points = points.shape[0]
    n_viewpoints = rotations.shape[0]

    points = transform_each(rotations, translations, points)

    points = points.reshape(-1, 3)
    observations = projection.project(points)
    observations = observations.reshape(n_viewpoints, n_points, 2)

    return observations


def generate_translations(rotations, points, offset=2.0):
    """
    Generate translations given rotations and 3D points such that
    depth > 0 for all points after transformation
    """
    n_viewpoints = rotations.shape[0]

    translations = np.empty((n_viewpoints, 3))
    for i in range(n_viewpoints):
        R = rotations[i]
        # convert all ponits to the camera coordinates
        P = np.dot(R, points.T).T
        # search the point which has the minimum z value
        argmin = np.argmin(P[:, 2])
        p = P[argmin]
        translations[i] = -p + np.array([0, 0, offset])
    return translations
