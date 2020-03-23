import numpy as np

# Kerl, Christian.
# "Odometry from rgb-d cameras for autonomous quadrocopters."
# Master's Thesis, Technical University (2012).


def calc_projection_jacobian(focal_lengths, P):
    fx, fy = focal_lengths

    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    z_squared = np.power(z, 2)
    a = x * y / z_squared

    JW = np.empty((P.shape[0], 2, 6))

    JW[:, 0, 0] = +fx / z
    JW[:, 0, 1] = 0
    JW[:, 0, 2] = -fx * x / z_squared
    JW[:, 0, 3] = -fx * a
    JW[:, 0, 4] = +fx * (1 + x * x / z_squared)
    JW[:, 0, 5] = -fx * y / z

    JW[:, 1, 0] = 0
    JW[:, 1, 1] = +fy / z
    JW[:, 1, 2] = -fy * y / z_squared
    JW[:, 1, 3] = -fy * (1 + y * y / z_squared)
    JW[:, 1, 4] = +fy * a
    JW[:, 1, 5] = +fy * x / z

    return JW


def calc_jacobian(focal_length, didx, didy, P0, P1):
    fx, fy = focal_length
    fgx, fgy = fx * didx, fy * didy

    x0, y0, z0 = P0[:, 0], P0[:, 1], P0[:, 2]
    x1, y1, z1 = P1[:, 0], P1[:, 1], P1[:, 2]

    z12 = z1 * z1
    x1y1 = x1 * y1

    J = np.empty((P0.shape[0], 6))
    J[:, 0] = fgx / z0
    J[:, 1] = fgy / z0
    J[:, 2] = -(fgx * x0 + fgy * y0) / (z0 * z0)
    J[:, 3] = -(fgx * x1y1 + fgy * (z12 + y1 * y1)) / z12
    J[:, 4] = (fgx * (z12 + x1 * x1) + fgy * x1y1) / z12
    J[:, 5] = (-fgx * y1 + fgy * x1) / z1
    return J


def calc_image_gradient(image):
    DY, DX = np.gradient(image)
    return DX, DY  # reverse the order
