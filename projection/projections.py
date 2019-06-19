from autograd import numpy as np


def pi(P):
    Z = P[:, [2]]
    XY = P[:, 0:2]
    return XY / Z


class BaseProjection(object):
    def __init__(self, camera_parameters):
        self.camera_parameters = camera_parameters

    def project(self, P):
        raise NotImplementedError()


class PerspectiveProjection(BaseProjection):
    def project(self, P):
        K = self.camera_parameters.matrix
        P = np.dot(K, P.T).T
        return pi(P)
