from autograd import numpy as np

from vitamine.so3 import rodrigues


class Pose(object):
    def __init__(self, R, t):
        self.R, self.t = R, t

    @staticmethod
    def identity():
        return Pose(np.identity(3), np.zeros(3))

    def __eq__(self, other):
        return (np.isclose(self.R, other.R).all() and
                np.isclose(self.t, other.t).all())
