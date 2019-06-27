import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rigid.rotation import tangent_so3


# Equation numbers are the ones in Multiple View Geometry

Z = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 0]
])


W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])


def project(p, K):
    q = np.dot(K, p)
    return q / q[2]


def solve_linear(A):
    # find x such that Ax = 0
    U, S, VH = np.linalg.svd(A)
    return VH[-1]


def estimate_fundamental(points0, points1):
    # Eq. 11.3
    assert(points0.shape == points1.shape)
    N = points0.shape[0]

    XA, YA = points0[:, 0], points0[:, 1]
    XB, YB = points1[:, 0], points1[:, 1]
    A = np.vstack((XB * XA, XB * YA, XB,
                   YB * XA, YB * YA, YB,
                   XA, YA, np.ones(N))).T
    f = solve_linear(A)
    F = f.reshape(3, 3)
    return F


def fundamental_to_essential(F, K0, K1=None):
    if K1 is None:
        K1 = K0
    return inv(K1).T.dot(F).dot(inv(K0))


# TODO compute multiple points
def structure_from_poses(K, R1, t1, point0, point1):
    def motion_matrix(R, t):
        T = np.empty((3, 4))
        T[0:3, 0:3] = R
        T[0:3, 3] = t
        return T

    R0, t0 = np.identity(3), np.zeros(3)

    P0 = np.dot(K, motion_matrix(R0, t0))
    P1 = np.dot(K, motion_matrix(R1, t1))

    x0, y0 = point0
    x1, y1 = point1

    A = np.vstack([
        x0 * P0[2] - P0[0],
        y0 * P0[2] - P0[1],
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1],
    ])
    x = solve_linear(A)
    return x / x[3]  # normalize so that x be a homogeneous vector


def projection_matrix(E, F, K):
    R, t = extract_pose(E)
    e = np.dot(K, t)  # project(t, K)
    S = tangent_so3(e.reshape(1, 3))[0]

    P = np.empty((3, 4))
    P[0:3, 0:3] = S.dot(F)
    P[0:3, 3] = e
    return P


def extract_poses(E):
    # Eq. 9.14
    U, s, VH = np.linalg.svd(E)
    s[2] = 0  # assume rank(E) == 2

    R1 = U.dot(W).dot(VH)
    R2 = U.dot(W.T).dot(VH)

    S = -U.dot(W).dot(np.diag(s)).dot(U.T)
    t = np.array([S[2, 1], S[0, 2], S[1, 0]])
    t1 = t
    t2 = -t
    return R1, R2, t1, t2


if __name__ == "__main__":
    from visualization import plot2d, plot3d

    K = np.identity(3)
    observations = np.load("observations.npy")
    points0, points1 = observations[0], observations[1]
    F = estimate_fundamental(points0, points1)
    E = fundamental_to_essential(F, K)
    plot2d(points0)
    plot2d(points1)
    R1, R2, t1, t2 = extract_poses(E)

    N = points0.shape[0]
    X = np.zeros((N, 4))
    for i in range(N):
        X[i] = structure_from_poses(K, R1, t1, points0[i], points1[i])

    plot3d(X)
    # fig = plt.figure()
    # ax = fig.add_subplot(221, projection='3d')
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    plt.show()
