import sympy
from sympy import BlockMatrix, Matrix, Symbol
from tadataka.codegen import generate_c_code


def MatrixSymbol(name, n, m):
    matrix = sympy.MatrixSymbol(name, n, m)
    for element in matrix:
        element._assumptions.update({
            'real': True,
            'commutative': True,
            'complex': True,
            'extended_real': True,
            'finite': True,
            'hermitian': True,
            'infinite': False,
            'imaginary': False
        })
    return matrix


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


def exp_so3_symbols(rotvec):
    epsilons = EPSILON * sympy.ones(3, 1)
    # add 'epsilons' to 'rotvec' directly
    # ZeroDivision occurs in grad calculation if we add EPSILON to 'theta'
    # in the denominator
    theta = Matrix(rotvec + epsilons).norm()
    K = tangent_so3(rotvec / theta)
    I = sympy.eye(3)
    return I + sympy.sin(theta) * K + (1-sympy.cos(theta)) * K * K


def transform_symbols(R, t, p):
    return R * p + t


def transform_project_symbols(rotvec, t, point):
    R = exp_so3_symbols(rotvec)
    q = transform_symbols(R, t, point)
    return Matrix(q[0:2]) / (q[2] + EPSILON)


def generate():
    rotvec = MatrixSymbol("rotvec", 3, 1)
    t = MatrixSymbol("t", 3, 1)
    pose = BlockMatrix([[rotvec], [t]])
    point = MatrixSymbol("point", 3, 1)

    x_symbols = transform_project_symbols(rotvec, t, point)

    generate_c_code("_transform_project", x_symbols,
                    prefix="tadataka/_transform_project/_transform_project")

    generate_c_code("_pose_jacobian", x_symbols.jacobian(pose),
                    prefix="tadataka/_transform_project/_pose_jacobian")

    generate_c_code("_point_jacobian", x_symbols.jacobian(point),
                    prefix="tadataka/_transform_project/_point_jacobian")

    generate_c_code("_exp_so3", exp_so3_symbols(rotvec),
                    prefix="tadataka/_transform_project/_exp_so3")


def exp_so3(rotvec):
    return exp_so3_(*rotvec)


def projection(pose, point):
    x_pred = projection_(*pose, *point)
    return x_pred.flatten()  # x_pred is a matrix of shape (2, 1)


def pose_jacobian(pose, point):
    return pose_jacobian_(*pose, *point)


def point_jacobian(pose, point):
    return point_jacobian_(*pose, *point)
