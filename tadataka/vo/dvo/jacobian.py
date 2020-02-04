import numpy as np

# Kerl, Christian.
# "Odometry from rgb-d cameras for autonomous quadrocopters."
# Master's Thesis, Technical University (2012).


def calc_projection_jacobian(camera_parameters, P):
    fx, fy = camera_parameters.focal_length

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


def calc_jacobian(camera_parameters, dx, dy, P):
    fx, fy = camera_parameters.focal_length
    fgx, fgy = fx * dx, fy * dy

    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    z2 = z * z  # element-wise z * z
    xy = x * y

    J = np.empty((P.shape[0], 6))
    J[:, 0] = fgx / z
    J[:, 1] = fgy / z
    J[:, 2] = -(fgx * x + fgy * y) / z2
    J[:, 3] = -fgx * xy / z2 - fgy * (1 + np.power(y / z, 2))
    J[:, 4] = fgx * (1 + np.power(x / z, 2)) + fgy * xy / z2
    J[:, 5] = (-fgx * y + fgy * x) / z
    return J


def calc_image_gradient(image):
    DY, DX = np.gradient(image)
    return DX, DY  # reverse the order
