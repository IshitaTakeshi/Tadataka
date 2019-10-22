from autograd import numpy as np


EPSILON = 1e-16


def is_rotation_matrix(R):
    assert(R.shape[0] == R.shape[1])
    I = np.identity(3)
    return (np.isclose(np.dot(R, R.T), I).all() and
            np.isclose(np.linalg.det(R), 1.0))


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

    assert(is_rotation_matrix(R))

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
    assert(RS.shape[1] == RS.shape[2] == 3)
    return np.vstack([log_so3(R) for R in RS])


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
    G = np.array([bases_so3 for i in range(N)])
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

    # Add EPSILON to avoid division by zero
    theta = np.linalg.norm(omegas + EPSILON, axis=1)
    # ZeroDivision occurs when calculating jacobian using autograd
    # if we add EPSILON to the denominator of this line
    # instead of the previous line
    K = tangent_so3(omegas / theta[:, np.newaxis])

    I = np.zeros((N, 3, 3))
    I[:, [0, 1, 2], [0, 1, 2]] = 1  # [np.identity(3) for i in range(N)]

    A = np.einsum('i,ijk->ijk', np.sin(theta), K)
    U = np.einsum('ijk,ikl->ijl', K, K)  # [dot(L, L) for L in K]
    B = np.einsum('i,ijk->ijk', 1-np.cos(theta), U)
    return I + A + B


def exp_so3(omega):
    assert(omega.shape == (3,))
    return rodrigues(np.atleast_2d(omega))[0]
