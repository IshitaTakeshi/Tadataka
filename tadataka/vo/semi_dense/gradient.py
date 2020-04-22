import numpy as np

from tadataka.interpolation import interpolation_


class GradientImage(object):
    def __init__(self, image_grad_x, image_grad_y):
        self.grad_x = image_grad_x
        self.grad_y = image_grad_y

    def __call__(self, u_key):
        u_key = u_key.astype(np.float64)
        gx = interpolation_(self.grad_x, u_key)
        gy = interpolation_(self.grad_y, u_key)
        return np.array([gx, gy])


def gradient1d(intensities):
    return intensities[1:] - intensities[:-1]


def calc_gradient(intensities):
    return np.linalg.norm(gradient1d(intensities))
