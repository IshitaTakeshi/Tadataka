from autograd import numpy as np


def affine_transform(A, b, X):
    return np.dot(A, X.T).T + b


class BaseTransform(object):
    def transform(self):
        raise NotImplementedError()


class AffineTransform(BaseTransform):
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def transform(self, X):
        return affine_transform(self.A, self.b, X)
