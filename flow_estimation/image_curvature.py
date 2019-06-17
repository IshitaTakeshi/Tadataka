import numpy as np
from skimage import data
from skimage.color import rgb2gray

from scipy import ndimage


def grad_x(image):
    return ndimage.sobel(image, axis=0, mode='constant')


def grad_y(image):
    return ndimage.sobel(image, axis=1, mode='constant')


def curvature(image):
    gx = grad_x(image)
    gy = grad_y(image)

    gxx = grad_x(gx)
    gxy = grad_y(gx)
    gyx = grad_x(gy)
    gyy = grad_y(gy)

    g2y = gy * gy
    g2x = gx * gx

    return g2y * gxx - gx * gy * gxy - gy * gx * gyx + g2x * gyy
