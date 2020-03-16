import numpy as np
import sympy
from sympy import Matrix, MatrixSymbol
from sympy.utilities.codegen import codegen


autowrap_backend = "cython"


def distort_(dist_coeffs, keypoint):
    x, y = keypoint
    k1, k2, p1, p2, k3 = dist_coeffs

    # arguments are sorted in sympy's autowrap
    x2 = x * x
    y2 = y * y
    xy = x * y
    r2 = x2 + y2
    r4 = r2 * r2
    r6 = r4 * r2
    kr = 1 + k1 * r2 + k2 * r4 + k3 * r6

    return Matrix([
        x * kr + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2),
        y * kr + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2)
    ])


dist_coeffs_ = MatrixSymbol('dist_coeffs', 5, 1)
keypoint_ = MatrixSymbol('keypoint', 2, 1)

distort_symbols_ = distort_(dist_coeffs_, keypoint_)
# jacobian should be 2x2
jacobian_symbols_ = distort_symbols_.jacobian(keypoint_)


def codegen_(name_expr, prefix):
    codegen(name_expr, prefix=prefix, language="C", to_files=True)


def generate():
    codegen_(("distort", distort_symbols_),
             prefix="tadataka/camera/_radtan_distort")

    codegen_(("distort_jacobian", jacobian_symbols_),
             prefix="tadataka/camera/_radtan_distort_jacobian")
