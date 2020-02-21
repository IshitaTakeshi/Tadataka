import numpy as np
from numpy.testing import assert_array_equal
from tadataka.gradient import grad_x, grad_y


A = np.arange(25).reshape(5, 5)
B = np.arange(49).reshape(7, 7)


def test_grad_x():
    GX = grad_x(A)
    assert_array_equal(GX[1:4, 1:4], 8)


def test_grad_y():
    GY = grad_y(A)
    assert_array_equal(GY[1:4, 1:4], 40)
