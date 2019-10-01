from autograd import numpy as np


EPSILON = 1e-16


def is_almost_zero(x):
    return np.isclose(x, 0)


def flip_omega(omega):
    # omega = [x, y, z] have to satisfy the condition below:
    # if x == 0:
    #     if y == 0:
    #         z == pi
    #     else:
    #         y > 0
    # else:
    #     x > 0

    x, y, z = omega
    if is_almost_zero(x) and is_almost_zero(y) and z < 0:
        # this handles only one case: [0, 0, -pi] -> [0, 0, pi]
        return -omega
    if is_almost_zero(x) and y < 0:
        # this condition can catch many cases
        # ex.
        # [0, -pi, 0] -> [0, pi, 0]
        # [0, -pi / sqrt(2), -pi / sqrt(2)] ->
        # [0,  pi / sqrt(2),  pi / sqrt(2)]
        return -omega
    if x < 0:
        # [-3 * pi / 5, 0,  4 * pi / 5] ->
        # [ 3 * pi / 5, 0, -4 * pi / 5]
        return -omega
    return omega


def log_so3(R):
    # Tomasi, Carlo. "Vector representation of rotations."
    # Computer Science 527 (2013).
    # https://www2.cs.duke.edu/courses/fall13/compsci527/notes/rodrigues.pdf

    c = (np.trace(R) - 1) / 2

    rho = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / 2

    s2 = np.sum(np.power(rho, 2))

    if is_almost_zero(s2) and np.isclose(c, 1):
        # diagonal elements are [1, 1, 1]
        return np.zeros(3)

    if is_almost_zero(s2) and np.isclose(c, -1):
        U = R + np.identity(3)
        argmax = np.argmax(np.linalg.norm(U, axis=0))
        v = U[:, argmax]
        u = v / np.linalg.norm(v)
        return flip_omega(np.pi * u)

    s = np.sqrt(s2)
    theta = np.arctan2(s, c)
    return theta * rho / s


def inv_rodrigues(RS):
    assert(np.all([np.isclose(np.dot(R.T, R), np.identity(3)).all() for R in RS]))
    assert(np.isclose(np.linalg.det(RS), 1.0).all())
    assert(RS.shape[1] == RS.shape[2] == 3)
    return np.vstack([log_so3(R) for R in RS])


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


def rodrigues(omegas):
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

    assert(omegas.shape[1] == 3)

    N = omegas.shape[0]

    theta = np.linalg.norm(omegas, axis=1)

    A = np.zeros((N, 3, 3))
    A[:, [0, 1, 2], [0, 1, 2]] = 1  # [np.eye(3) for i in range(N)]

    if np.all(theta == 0):
        return A

    # Add EPSILON to avoid division by zero
    K = tangent_so3(omegas / (theta[:, np.newaxis] + EPSILON))

    B = np.einsum('i,ijk->ijk', np.sin(theta), K)
    C = np.einsum('ijk,ikl->ijl', K, K)  # [dot(L, L) for L in K]
    C = np.einsum('i,ijk->ijk', 1-np.cos(theta), C)
    return A + B + C
