import numpy as np
import sympy
from sympy import Matrix
from sympy.utilities.autowrap import autowrap
from sympy.utilities.codegen import codegen


autowrap_backend = "cython"


def distort_(k1, k2, k3, p1, p2, x, y):
    # arguments are sorted in sympy's autowrap
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r2 * r2 * r2
    kr = 1 + k1 * r2 + k2 * r4 + k3 * r6

    return [
        x * kr + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * kr + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
    ]


x_, y_ = sympy.symbols('x y', real=True)
k1_, k2_, k3_, p1_, p2_ = sympy.symbols('k1 k2 k3 p1 p2', real=True)

distort_symbols_ = Matrix(distort_(k1_, k2_, k3_, p1_, p2_, x_, y_))
jacobian_symbols_ = distort_symbols_.jacobian(Matrix([x_, y_]))

distort_jacobian_ = autowrap(jacobian_symbols_, backend=autowrap_backend)


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


# (c_name, c_code), (h_name, h_code) = codegen(
#     ("jacobian", jacobian_symbols_),
#     language="C", prefix="radtan_jacobian"
# )
# print(c_name)
# print(c_code)
# print(h_name)
# print(h_code)
