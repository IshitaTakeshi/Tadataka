from autograd import numpy as np


EPSILON = 1e-8


def log_so3(RS):
    # FIXME
    # this implementation cannot calculate omega correctly
    # in the case if R is symmetric but not identity, such as
    #
    #     [-1 0 0]
    # R = [0 -1 0]
    #     [0 0 0]
    #
    # The corresponding omega = [0 0 0] and theta = pi but
    # exp(omega * theta) = identity
    #
    # This happens because omega is at singularity

    assert(RS.shape[1] == RS.shape[2] == 3)
    N = RS.shape[0]

    traces = np.trace(RS, axis1=1, axis2=2)
    thetas = np.arccos((traces - 1) / 2)

    mask = np.abs(thetas) > EPSILON

    omegas = np.empty((N, 3))

    omegas[np.logical_not(mask), :] = 0.

    omegas[mask, 0] = RS[mask, 2, 1] - RS[mask, 1, 2]
    omegas[mask, 1] = RS[mask, 0, 2] - RS[mask, 2, 0]
    omegas[mask, 2] = RS[mask, 1, 0] - RS[mask, 0, 1]

    sins = np.sin(thetas[mask]).reshape(-1, 1)  # align shape as omegas
    omegas[mask] = omegas[mask] / (2 * sins)

    return omegas, thetas


def tangent_so3(V):
    """
    V: np.ndarray
        Array of se3 elements.
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
    theta[theta == 0] = 1  # avoid division by zero

    V = V / theta[:, np.newaxis]
    K = tangent_so3(V)

    A = np.zeros((N, 3, 3))
    A[:, [0, 1, 2], [0, 1, 2]] = 1  # [np.eye(3) for i in range(N)]

    B = np.einsum('i,ijk->ijk', np.sin(theta), K)
    C = np.einsum('ijk,ikl->ijl', K, K)  # [dot(L, L) for L in K]
    C = np.einsum('i,ijk->ijk', 1-np.cos(theta), C)
    return A + B + C
