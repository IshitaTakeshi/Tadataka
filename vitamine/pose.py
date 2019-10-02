from autograd import numpy as np

from vitamine.so3 import exp_so3, log_so3


class Pose(object):
    def __init__(self, R_or_omega, t):
        if np.ndim(R_or_omega) == 1:
            self.omega = R_or_omega
        elif np.ndim(R_or_omega) == 2:
            self.omega = log_so3(R_or_omega)

        self.t = t

    @property
    def R(self):
        return exp_so3(self.omega)

    def __str__(self):
        with np.printoptions(precision=3, suppress=True):
            return "omega = " + str(self.omega)  + "   t = " + str(self.t)

    @staticmethod
    def identity():
        return Pose(np.zeros(3), np.zeros(3))

    def __eq__(self, other):
        return (np.isclose(self.omega, other.omega).all() and
                np.isclose(self.t, other.t).all())
