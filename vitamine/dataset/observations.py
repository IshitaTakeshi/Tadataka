from autograd import numpy as np

from vitamine.so3 import rodrigues
from vitamine.rigid.transformation import transform_all


def generate_observations(rotations, translations, points, projection):
    assert(points.shape[1] == 3)
    assert(rotations.shape[0] == translations.shape[0])
    assert(translations.shape[1] == 3)
    assert(rotations.shape[1:3] == (3, 3))

    n_points = points.shape[0]
    n_viewpoints = rotations.shape[0]

    points = transform_all(rotations, translations, points)

    positive_depth_mask = points[:, :, 2] > 0

    observations = projection.compute(points.reshape(-1, 3))
    observations = observations.reshape(*points.shape[0:2], 2)

    return observations, positive_depth_mask


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
