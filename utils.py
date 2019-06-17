from autograd import numpy as np


def from_2d(x):
    return x.flatten()


def to_2d(x):
    return x.reshape(-1, 2)


def affine_matrix(A, b):
    W = np.identity(3)
    W[0:2, 0:2] = A
    W[0:2, 2] = b
    return W
