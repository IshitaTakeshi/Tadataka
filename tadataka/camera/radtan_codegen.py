import sympy
from sympy import Matrix
from sympy.utilities.autowrap import autowrap


autowrap_backend = "cython"


def distort_symbols(k1, k2, k3, p1, p2, x, y):
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r2 * r2 * r2
    kr = 1 + k1 * r2 + k2 * r4 + k3 * r6

    return Matrix([
        x * kr + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * kr + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
    ])


x_, y_ = sympy.symbols('x y', real=True)
k1_, k2_, k3_, p1_, p2_ = sympy.symbols('k1 k2 k3 p1 p2', real=True)

distort_symbols_ = distort_symbols(k1_, k2_, k3_, p1_, p2_, x_, y_)

distort_ = autowrap(distort_symbols_, backend=autowrap_backend)
distort_jacobian_ = autowrap(distort_symbols_.jacobian(Matrix([x_, y_])),
                             backend=autowrap_backend)


def distort(keypoint, dist_coeffs):
    x, y = keypoint
    k1, k2, p1, p2, k3 = dist_coeffs
    return distort_(k1, k2, k3, p1, p2, x, y)


def distort_jacobian(keypoint, dist_coeffs):
    x, y = keypoint
    k1, k2, p1, p2, k3 = dist_coeffs
    return distort_jacobian_(k1, k2, k3, p1, p2, x, y)
