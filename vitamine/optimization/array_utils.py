from vitamine.optimization.functions import Function


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, x):
        return x.reshape(self.shape)


class Flatten(object):
    def compute(self, x):
        return x.flatten()
