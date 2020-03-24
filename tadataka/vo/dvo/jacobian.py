import numpy as np

# Kerl, Christian.
# "Odometry from rgb-d cameras for autonomous quadrocopters."
# Master's Thesis, Technical University (2012).


def calc_jacobian(focal_length, didx, didy, P):
    fx, fy = focal_length
    fgx, fgy = fx * didx, fy * didy

    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    z2 = z * z
    xy = x * y

    J = np.empty((P.shape[0], 6))
    J[:, 0] = fgx / z
    J[:, 1] = fgy / z
    J[:, 2] = -(fgx * x + fgy * y) / (z * z)
    J[:, 3] = -(fgx * xy + fgy * (z2 + y * y)) / z2
    J[:, 4] = (fgx * (z2 + x * x) + fgy * xy) / z2
    J[:, 5] = (-fgx * y + fgy * x) / z
    return J


def calc_image_gradient(image):
    DY, DX = np.gradient(image)
    return DX, DY  # reverse the order
