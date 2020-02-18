import numpy as np
from scipy.spatial.transform import Rotation


EPSILON = 1e-16


def is_rotation_matrix(R):
    assert(R.shape[0] == R.shape[1])
    I = np.identity(3)
    return (np.isclose(np.dot(R, R.T), I).all() and
            np.isclose(np.linalg.det(R), 1.0))


bases_so3 = np.array([
    [[0, 0, 0],
     [0, 0, -1],
     [0, 1, 0]],
    [[0, 0, 1],
     [0, 0, 0],
     [-1, 0, 0]],
    [[0, -1, 0],
     [1, 0, 0],
     [0, 0, 0]]
])


def tangent_so3(v):
    """
    v: np.ndarray
        se3 elements
    """
    #
    #        [0  0   0]            [ 0  0  1]            [0  -1  0]
    # v[0] * [0  0  -1]  +  v[1] * [ 0  0  0]  +  v[2] * [1   0  0]
    #        [0  1   0]            [-1  0  0]            [0   0  0]

    return np.einsum('jkl,j->kl', bases_so3, v)


def exp_so3(rotvec):
    return Rotation.from_rotvec(rotvec).as_matrix()


def log_so3(R):
    return Rotation.from_matrix(R).as_rotvec()
