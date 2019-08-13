from vitamine.matrix import affine_transform


class BaseTransform(object):
    def transform(self):
        raise NotImplementedError()


class AffineTransform(BaseTransform):
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def transform(self, X):
        return affine_transform(X, self.A, self.b)
