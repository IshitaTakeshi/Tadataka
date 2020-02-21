import numpy as np
from numpy.testing import assert_almost_equal

from tadataka.rigid_transform import Transform
from tadataka.vo.semi_dense.depth import (
    step_size_ratio, depth_search_range, epipolar_search_range,
    GradientImage)


def test_step_size_ratio():
    R = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    t = np.array([-1, 4, -4])
    x_key = np.array([4, -3])
    inv_depth = 0.25

    assert_almost_equal(
        step_size_ratio(Transform(R, t), x_key, inv_depth),
        0.25 / (1 / -20)
    )


def test_depth_search_range():
    # invert and reverse the order
    assert_almost_equal(depth_search_range((0.25, 0.8)), (1.25, 4.0))
    assert_almost_equal(depth_search_range((0.10, 0.5)), (2.0, 10.0))


def test_epipolar_search_range():
    R = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    t = np.array([-1, 4, 2])
    x_key = np.array([-1, 4])
    depth_range = (4, 8)

    transform = Transform(R, t)
    x_ref_min, x_ref_max = epipolar_search_range(transform, x_key, depth_range)

    # 4 * [1  4  1] + [-1  4  2] = [4-1  16+4  4+2]
    # 8 * [1  4  1] + [-1  4  2] = [8-1  32+4  8+2]
    assert_almost_equal(x_ref_min, [3 / 6, 20 / 6])
    assert_almost_equal(x_ref_max, [7 / 10, 36 / 10])


def test_gradient_image():
    width, height = 6, 4
    image_grad_x = np.arange(0, 24).reshape(height, width)
    image_grad_y = np.arange(24, 48).reshape(height, width)
    gradient_image = GradientImage(image_grad_x, image_grad_y)

    u_key = [4.3, 2.1]
    gx, gy = gradient_image(u_key)

    u, v = 4, 2

    expected_x = (0.7 * 0.9 * image_grad_x[v, u] +
                  0.3 * 0.9 * image_grad_x[v, u+1] +
                  0.7 * 0.1 * image_grad_x[v+1, u] +
                  0.3 * 0.1 * image_grad_x[v+1, u+1])
    assert_almost_equal(gx, expected_x)

    expected_y = (0.7 * 0.9 * image_grad_y[v, u] +
                  0.3 * 0.9 * image_grad_y[v, u+1] +
                  0.7 * 0.1 * image_grad_y[v+1, u] +
                  0.3 * 0.1 * image_grad_y[v+1, u+1])
    assert_almost_equal(gy, expected_y)
