from numpy.testing import assert_array_almost_equal
import numpy as np


def random_rotation_matrix_3d():
    """Generate a random matrix in :math:`\mathbb{SO}(3)`"""
    A = np.random.uniform(-1, 1, (3, 3))  # Generate random matrix A
    Q = np.dot(A, A.T)  # Q is a symmetric matrix
    # Q is decomposed into RDR^T, where R is a rotation matrix
    R = np.linalg.svd(Q)[0]
    return R


def random_vector_3d(scale=1.0):
    """
    Generate a random 3D vector :math:`\mathbf{v}` such that
        :math:`\mathbf{v}_{i} \in [-s, s),\,i=1,2,3`
    """
    v = np.random.uniform(-1, 1, size=3)
    v = v / np.linalg.norm(v)
    return scale * v


def calculate_rotation(X, Y):
    S = np.dot(X.T, Y)

    U, _, VT = np.linalg.svd(S)  # S = U * Sigma * VT
    V = VT.T
    return np.dot(V, U.T)


def calculate_scaling(X, Y, R):
    n = np.sum(np.dot(np.dot(y, R), x) for x, y in zip(X, Y))
    d = np.sum(X * X)  # equivalent to sum([dot(x, x) for x in X])
    return n / d


def calculate_translation(s, R, p, q):
    return q - s * np.dot(R, p)


class LeastSquaresRigidMotion(object):
    """
    For each element in :math:`P = \{\mathbf{p}_i\}`
    and the corresponding element in :math:`Q = \{\mathbf{q}_i\}`,
    calculate the transformation which minimizes the error

    .. math::
        E(P, Q) = \sum_{i} ||s R \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i||^2

    where :math:`s`, :math:`R`, :math:`\mathbf{t}` are scaling factor,
    rotation matrix and translation vector respectively.

    Examples:

    >>> s, R, t = LeastSquaresRigidMotion(P, Q).solve()
    >>> P = transform(s, R, t, P)

    See :cite:`zinsser2005point` for the detailed method.

        .. bibliography:: refs.bib
    """

    def __init__(self, P: np.ndarray, Q: np.ndarray):
        """
        Args:
            P: Set of points of shape (n_image_points, n_channels)
                to be transformed
            Q: Set of points of shape (n_image_points, n_channels)
                to be used as a reference
        """

        if P.shape != Q.shape:
            raise ValueError("P and Q must be the same shape")

        self.n_features = P.shape[1]
        self.P = P
        self.Q = Q

    def solve(self):
        """
        Calculate (:math:`s`, :math:`R`, :math:`\mathbf{t}`)

        Returns:
            tuple: (s, R, t) where
                - :math:`s`: Scaling coefficient
                - :math:`R`: Rotation matrix
                - :math:`\mathbf{t}`: translation vector
        """

        mean_p = np.mean(self.P, axis=0)
        mean_q = np.mean(self.Q, axis=0)

        X = self.P - mean_p
        Y = self.Q - mean_q

        R = calculate_rotation(X, Y)
        s = calculate_scaling(X, Y, R)
        t = calculate_translation(s, R, mean_p, mean_q)

        return s, R, t


def transform(s: float, R: np.ndarray,
              t: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Transform each point :math:`\mathbf{p} \in P` into
    :math:`\mathbf{q} = sR\mathbf{p} + \mathbf{t}` where

    - :math:`s`: Scaling factor
    - :math:`R`: Rotation matrix
    - :math:`\mathbf{t}`: Translation vector

    Args:
        s: Scaling factor
        R: Rotation matrix
        t: Translation vector
        P: Points to be transformed of shape (n_image_points, n_channels)

    Returns:
        Transformed vector
    """

    if np.ndim(t) == 1:
        t = t.reshape(-1, 1)  # align the dimension

    P = s * np.dot(R, P.T) + t

    return P.T
