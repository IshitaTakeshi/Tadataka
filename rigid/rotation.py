from autograd import numpy as np


def tangent_so3(V):
    """
    V: np.ndarray
        Array of se3 elements
    """
    # Generate skew matrices by calculating below for
    # all elements along the 0-th axis of V
    #
    #        [0  0   0]            [ 0  0  1]            [0  -1  0]
    # V[0] * [0  0  -1]  +  V[1] * [ 0  0  0]  +  V[2] * [1   0  0]
    #        [0  1   0]            [-1  0  0]            [0   0  0]

    N = V.shape[0]

    def bases(indices):
        """
        Generate bases of so(3)
        """
        G = np.zeros((N, len(indices), 3, 3))
        for k, (i, j) in enumerate(indices):
            G[:, k, i, j] = 1
            G[:, k, j, i] = -1
        return G

    indices = [
        [2, 1],
        [0, 2],
        [1, 0],
    ]

    G = bases(indices)
    return np.einsum('ijkl,ij->ikl', G, V)


def rodrigues(V):
    """
    see
    https://docs.opencv.org/2.4/modules/calib3d/doc/
    camera_calibration_and_3d_reconstruction.html#rodrigues

    .. codeblock:
        theta = np.linalg.norm(r)
        r = r / theta
        K = adj_so3(r)
        I = np.eye(3, 3)
        return I + np.sin(theta) * K + (1-np.cos(theta)) * np.dot(K, K)

    I + sin(theta) * K + (1-cos(theta)) * dot(K, K) is equivalent to
    cos(theta) * I + (1-cos(theta)) * outer(r, r) + sin(theta) * K
    """

    assert(V.shape[1] == 3)

    N = V.shape[0]

    theta = np.linalg.norm(V, axis=1)
    V = V / theta[:, np.newaxis]
    K = tangent_so3(V)

    A = np.zeros((N, 3, 3))
    A[:, [0, 1, 2], [0, 1, 2]] = 1  # [np.eye(3) for i in range(N)]

    B = np.einsum('i,ijk->ijk', np.sin(theta), K)
    C = np.einsum('ijk,ikl->ijl', K, K)  # [dot(L, L) for L in K]
    C = np.einsum('i,ijk->ijk', 1-np.cos(theta), C)
    return A + B + C
