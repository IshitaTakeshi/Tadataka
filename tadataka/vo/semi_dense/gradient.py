import numpy as np

from tadataka.interpolation import interpolation


class GradientImage(object):
    def __init__(self, image_grad_x, image_grad_y):
        self.grad_x = image_grad_x
        self.grad_y = image_grad_y

    def __call__(self, u_key):
        u_key = u_key.astype(np.float64)
        gx = interpolation(self.grad_x, u_key)
        gy = interpolation(self.grad_y, u_key)
        return np.array([gx, gy])
