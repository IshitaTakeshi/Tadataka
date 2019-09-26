from autograd import numpy as np


def inv_motion_matrix(T):
    R, t = get_rotation_translation(T)
    return motion_matrix(R.T, -np.dot(R.T, t))


def get_rotation_translation(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    return R, t


def motion_matrix(R, t):
    T = np.empty((3, 4))
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def homogeneous_matrix(A, b):
    """
    Example:
        >>> A = np.array([[1, 2], [3, 4]])
        >>> b = np.array([5, 6])
        >>> homogeneous_matrix(A, b)
        array([[1., 2., 5.],
               [3., 4., 6.],
               [0., 0., 1.]])
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("'A' must be a square matrix")

    if A.shape[0] != b.shape[0]:
        raise ValueError("Number of rows of 'A' must match "
                         "the number of elements of 'b'")
    d = A.shape[0]

    W = np.identity(d+1)
    W[0:d, 0:d] = A
    W[0:d, d] = b
    return W


def affine_transform(X, A, b):
    return np.dot(A, X.T).T + b


def to_homogeneous(X):
    """
    Args:
        X: coordinates of shape (n_points, dim)
    Returns:
        Homogeneous coordinates of shape (n_points, dim + 1)
    """
    ones = np.ones((X.shape[0], 1))
    return np.hstack((X, ones))


def motion_matrix(R, t):
    T = np.empty((4, 4))
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    T[3, 0:3] = 0
    T[3, 3] = 1
    return T


def from_homogeneous(X):
    d = X.shape[1] - 1
    return X[:, 0:d]


def homogeneous_transformation(X, T):
    """
    X : image coordinates of shape (n_points, d)
    T : (d + 1) x (d + 1) transformation matrix
    """

    X = to_homogeneous(X)
    Y = np.dot(T, X.T).T
    return from_homogeneous(Y)


def solve_linear(A):
    # find x such that
    # Ax = 0  if A.shape[0] < A.shape[1]
    # min ||Ax|| otherwise
    # x is a vector in the kernel space of A
    U, S, VH = np.linalg.svd(A)
    return VH[-1]
