import numpy as np
import sympy
from sympy import Matrix
from sympy.utilities.codegen import codegen


autowrap_backend = "cython"


def distort_(k1, k2, k3, p1, p2, x, y):
    # arguments are sorted in sympy's autowrap
    x2 = x * x
    y2 = y * y
    xy = x * y
    r2 = x2 + y2
    r4 = r2 * r2
    r6 = r4 * r2
    kr = 1 + k1 * r2 + k2 * r4 + k3 * r6

    return [
        x * kr + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2),
        y * kr + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2)
    ]


x_, y_ = sympy.symbols('x y', real=True)
k1_, k2_, k3_, p1_, p2_ = sympy.symbols('k1 k2 k3 p1 p2', real=True)

distort_symbols_ = Matrix(distort_(k1_, k2_, k3_, p1_, p2_, x_, y_))
# jacobian should be 2x2
jacobian_symbols_ = distort_symbols_.jacobian(Matrix([x_, y_]))


def distort(keypoints, dist_coeffs):
    xs, ys = keypoints[:, 0], keypoints[:, 1]
    # sort arguments
    k1, k2, p1, p2, k3 = dist_coeffs

    xs, ys = distort_(k1, k2, k3, p1, p2, xs, ys)
    return np.column_stack((xs, ys))


def distort_jacobian(keypoint, dist_coeffs):
    x, y = keypoint
    # sort arguments
    k1, k2, p1, p2, k3 = dist_coeffs
    return distort_jacobian_(k1, k2, k3, p1, p2, x, y)


def generate():
    codegen(("distort_jacobian", jacobian_symbols_),
            language="C", to_files=True,
            prefix="tadataka/camera/_radtan_distort_jacobian")
