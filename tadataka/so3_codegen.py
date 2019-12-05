import sympy
from sympy import Matrix
from sympy.utilities.autowrap import autowrap


autowrap_backend = "cython"


so3_base0 = Matrix([
    [0, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
])

so3_base1 = Matrix([
    [0, 0, 1],
    [0, 0, 0],
    [-1, 0, 0]
])

so3_base2 = Matrix([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 0]
])


def tangent_so3(v):
    return v[0] * so3_base0 + v[1] * so3_base1 + v[2] * so3_base2


EPSILON = 1e-16


def exp_so3_symbols(omega):
    epsilons = EPSILON * sympy.ones(1, 3)
    # add 'epsilons' to 'omega' directly
    # ZeroDivision occurs in grad calculation if we add EPSILON to 'theta'
    theta = (omega + epsilons).norm()
    K = tangent_so3(omega / theta)
    I = sympy.eye(3)
    return I + sympy.sin(theta) * K + (1-sympy.cos(theta)) * K * K


def transform_symbols(R, t, P):
    return (R * P.T).T + t


def projection_symbols(pose, point):
    omega, t = Matrix([pose[0:3]]), Matrix([pose[3:6]])
    R = exp_so3_symbols(omega)
    p = transform_symbols(R, t, point)
    return Matrix(p[0:2]) / (p[2] + EPSILON)


a, b, c, d, e, f = sympy.symbols('a:f', real=True)
x, y, z = sympy.symbols('x:z', real=True)

omega, t = Matrix([[a, b, c]]), Matrix([[d, e, f]])
pose = Matrix([*omega, *t])
point = Matrix([[x, y, z]])

x_symbols = projection_symbols(pose, point)
projection_ = autowrap(x_symbols, backend=autowrap_backend)

pose_jacobian_ = autowrap(x_symbols.jacobian(pose), backend=autowrap_backend)
point_jacobian_ = autowrap(x_symbols.jacobian(point), backend=autowrap_backend)

exp_so3_ = autowrap(exp_so3_symbols(omega), backend=autowrap_backend)


def exp_so3(omega):
    return exp_so3_(*omega)


def projection(pose, point):
    x_pred = projection_(*pose, *point)
    return x_pred.flatten()  # x_pred is a matrix of shape (2, 1)


def pose_jacobian(pose, point):
    return pose_jacobian_(*pose, *point)


def point_jacobian(pose, point):
    return point_jacobian_(*pose, *point)
