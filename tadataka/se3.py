import numpy as np
from tadataka.so3 import exp_so3, log_so3, tangent_so3


EPSILON = 1e-16


def normalize(omega):
    theta = np.linalg.norm(omega)
    if theta == 0:
        return np.zeros(len(omega)), 0
    return omega / theta, theta


def exp_se3(xi):
    v, rotvec = xi[:3], xi[3:]

    I = np.eye(3)
    R = exp_so3(rotvec)

    omega, theta = normalize(rotvec)
    K = tangent_so3(omega)

    # NOTE K is computed from the normalized rotvec

    if theta < EPSILON:  # since theta = norm(omega) >= 0
        V = I + K * theta / 2 + np.dot(K, K) * pow(theta, 2) / 6
    else:
        V = (I + (1 - np.cos(theta)) / theta * K +
             (theta - np.sin(theta)) / theta * np.dot(K, K))

    G = np.empty((4, 4))
    G[0:3, 0:3] = R
    G[0:3, 3] = np.dot(V, v)
    G[3, 0:3] = 0
    G[3, 3] = 1
    return G


def log_se3(G):
    # Gallier, Jean, and Dianna Xu. "Computing exponentials of skew-symmetric
    # matrices and logarithms of orthogonal matrices." International Journal of
    # Robotics and Automation 18.1 (2003): 10-20.

    R = G[0:3, 0:3]
    t = G[0:3, 3]

    rotvec = log_so3(R)
    omega, theta = normalize(rotvec)

    if theta == 0:
        # v == t if theta == 0
        return np.concatenate((t, omega * theta))

    K = tangent_so3(omega)
    I = np.eye(3)
    alpha = -theta / 2
    beta = 1 - theta * np.sin(theta) / (2 * (1 - np.cos(theta)))
    V_inv = I + alpha * K + beta * np.dot(K, K)
    v = V_inv.dot(t)
    return np.concatenate((v, omega * theta))


def get_rotation(G):
    return G[0:3, 0:3]


def get_translation(G):
    return G[0:3, 3]
