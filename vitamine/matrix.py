from autograd import numpy as np


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
