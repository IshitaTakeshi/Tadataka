from autograd import numpy as np


# def onehot_3x3(i, j):
#     K = np.zeros((3, 3))
#     K[i, j] = 1
#     return K
#
#
# def tangent_so3(v):
#     # What we want is the skew matrix in the form
#     # [[0, -z, y],
#     #  [z, 0, -x],
#     #  [-y, x, 0]]
#     # however, autograd doesn't support item assinments so what
#     # we have to write like below
#     return onehot_3x3(2, 1) * v[0]  # x
#          + onehot_3x3(1, 2) * -v[0]  # -x
#          + onehot_3x3(0, 2) * v[1]  # y
#          + onehot_3x3(2, 0) * -v[1]  # -y
#          + onehot_3x3(1, 0) * v[2]  # z
#          + onehot_3x3(0, 1) * -v[2]  # -z



def tangents_so3(V):
    N = V.shape[0]

    def onehots(indices):
        X = np.zeros((N, len(indices), 3, 3))
        for i, (j, k) in enumerate(indices):
            X[i, :, j, k] = 1
        return X

    indices = [
        [2, 1],
        [1, 2],
        [0, 2],
        [2, 0],
        [1, 0],
        [0, 1]
    ]

    # W.shape == (N, 6)
    W = np.vstack((
        V[:, 0],  # x
        -V[:, 0],  # -x
        V[:, 1],  # y
        -V[:, 1],  # -y
        V[:, 2],  # z
        -V[:, 2],  # -z
    )).T

    X = onehots(indices)
    return np.einsum('ijkl,ij->ikl', X, W)


def rodrigues(V):
    """
    # see
    # https://docs.opencv.org/2.4/modules/calib3d/doc/
    # camera_calibration_and_3d_reconstruction.html#rodrigues

    .. codeblock:
        theta = np.linalg.norm(r)
        r = r / theta
        K = adj_so3(r)
        I = np.eye(3, 3)
        return I + np.sin(theta) * K + (1-np.cos(theta)) * np.dot(K, K)

    # I + sin(theta) * K + (1-cos(theta)) * dot(K, K) is equivalent to
    # cos(theta) * I + (1-cos(theta)) * outer(r, r) + sin(theta) * K
    """

    print(V.shape)
    assert(V.shape[1] == 3)

    N = V.shape[0]

    # HACK this can be accelerated by calculating (V * V).sum(axis=1)
    theta = np.linalg.norm(V, axis=1)
    V = V / theta[:, np.newaxis]
    K = tangents_so3(V)

    A = np.zeros((N, 3, 3))
    A[:, [0, 1, 2], [0, 1, 2]] = 1  # [np.eye(3) for i in range(N)]

    B = np.einsum('i,ijk->ijk', np.sin(theta), K)
    C = np.einsum('ijk,ikl->ijl', K, K)  # [dot(L, L) for L in K]
    C = np.einsum('i,ijk->ijk', 1-np.cos(theta), C)
    return A + B + C
