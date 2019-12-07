import numpy as np

from tadataka.so3 import rodrigues
from tadataka.rigid_transform import transform_all


def generate_translations(rotations, points, depth_margin=2.0):
    """
    Generate translations given rotations and 3D points such that
    depth > 0 for all points after transformation
    """
    n_viewpoints = rotations.shape[0]

    translations = np.empty((n_viewpoints, 3))
    offset = np.array([0, 0, depth_margin])
    for i in range(n_viewpoints):
        R = rotations[i]
        # convert all ponits to the camera coordinates
        P = np.dot(R, points.T).T
        # search the point which has the minimum z value
        argmin = np.argmin(P[:, 2])
        p = P[argmin]
        translations[i] = -p + offset
    return translations
