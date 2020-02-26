import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from skimage.color import rgb2gray

from tadataka.pose import calc_relative_pose
from tadataka.dataset import NewTsukubaDataset
from tadataka.coordinates import image_coordinates

from tadataka.vo.semi_dense.semi_dense import (
    step_size_ratio, depth_search_range, epipolar_search_range,
    GradientImage, InverseDepthEstimator, InverseDepthSearchRange)
from tadataka.rigid_transform import Transform

from tests.dataset.path import new_tsukuba


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


def test_inverse_depth_search_range():
    inv_depth_range = InverseDepthSearchRange(
        min_inv_depth=0.02, max_inv_depth=20.0, factor=2.0)
    assert_equal(inv_depth_range(1.0, 0.2), (0.6, 1.4))
    assert_equal(inv_depth_range(1.0, 1.5), (0.02, 4.0))
    assert_equal(inv_depth_range(18.0, 1.5), (15.0, 20.0))

    with pytest.raises(AssertionError):
        # min_inv_depth must be positive
        InverseDepthSearchRange(min_inv_depth=-0.01, max_inv_depth=1.0)

    with pytest.raises(AssertionError):
        # max_inv_depth must be greater than min_inv_depth
        InverseDepthSearchRange(min_inv_depth=1.0, max_inv_depth=0.9)


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


def test_inverse_depth_estimator():
    prior_inv_depth = 0.15
    variance = 0.05
    dataset = NewTsukubaDataset(new_tsukuba)
    keyframe, refframe = dataset[0]
    pose_key_to_ref = calc_relative_pose(keyframe.pose, refframe.pose)
    sigma_i = 0.01
    sigma_l = 0.02
    estimator = InverseDepthEstimator(
        pose_key_to_ref, rgb2gray(keyframe.image), rgb2gray(refframe.image),
        keyframe.camera_model, refframe.camera_model,
        sigma_i, sigma_l,
        step_size_ref=0.01
    )
    u_key = [320, 200]
    x_key = keyframe.camera_model.normalize(np.atleast_2d(u_key))[0]

    inv_depth, variance = estimator(x_key, u_key, prior_inv_depth, variance)
    # print(keyframe.depth_map[u_key[1], u_key[0]])
    # print(1 / inv_depth, variance)
