import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
                           assert_array_almost_equal)
import pytest
from skimage.color import rgb2gray
from scipy.spatial.transform import Rotation

from tadataka.pose import calc_relative_pose
from tadataka.dataset import NewTsukubaDataset
from tadataka.coordinates import image_coordinates

from tadataka.vo.semi_dense.semi_dense import (
    step_size_ratio, depth_search_range, epipolar_search_range_,
    GradientImage, InverseDepthEstimator, InverseDepthSearchRange)

from tests.dataset.path import new_tsukuba


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
    R_key = np.identity(3)
    t_key = np.array([0, 0, -2])

    R_ref = Rotation.from_rotvec([0, -np.pi/2, 0]).as_matrix()
    t_ref = np.array([1, 0, 0])

    x_key = np.zeros(2)
    depth_range = [1, 3]
    x_ref_min, x_ref_max = epipolar_search_range_(
        (R_key, t_key), (R_ref, t_ref),
        x_key, depth_range
    )

    assert_array_almost_equal(x_ref_min, [-1, 0])
    assert_array_almost_equal(x_ref_max, [1, 0])


def test_gradient_image():
    width, height = 6, 4
    image_grad_x = np.arange(0, 24).reshape(height, width)
    image_grad_y = np.arange(24, 48).reshape(height, width)
    gradient_image = GradientImage(image_grad_x.astype(np.float64),
                                   image_grad_y.astype(np.float64))

    u_key = np.array([4.3, 2.1])
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
