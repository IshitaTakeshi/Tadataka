import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
                           assert_array_almost_equal)
from scipy.spatial.transform import Rotation
from skimage.color import rgb2gray

from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tadataka.camera import CameraModel, CameraParameters
from tadataka.vo.semi_dense.hypothesis import Hypothesis
from tadataka.vo.semi_dense.frame import Frame
from tadataka.vo.semi_dense.semi_dense import InvDepthEstimator
from tadataka.vo.semi_dense.gradient import GradientImage
from tadataka.gradient import grad_x, grad_y
from tadataka.dataset import NewTsukubaDataset
from tadataka.coordinates import image_coordinates
from tadataka.vo.semi_dense.common import invert_depth

from tests.dataset.path import new_tsukuba


def test_inv_depth_estimator():
    dataset = NewTsukubaDataset(new_tsukuba)
    keyframe, refframe = dataset[0]

    camer_model_ref = refframe.camera_model
    camer_model_key = keyframe.camera_model

    image_ref = rgb2gray(refframe.image)
    image_key = rgb2gray(keyframe.image)

    inv_depth_search_range = (0.001, 10.0)
    sigma_i = 0.01
    sigma_l = 0.01
    step_size_ref = 0.01
    min_gradient = 0.2

    estimator = InvDepthEstimator(camer_model_key, image_key,
                                  inv_depth_search_range,
                                  sigma_i, sigma_l, step_size_ref, min_gradient)
    pose_wk = keyframe.pose
    pose_wr = refframe.pose
    pose_rk = pose_wr.inv() * pose_wk
    T_rk = pose_rk.T

    def estimate(u_key, prior):
        return estimator(camer_model_ref, image_ref, T_rk, u_key, prior)

    u_key = np.array([110, 400])
    prior = Hypothesis(-0.1, 10.0)
    hypothesis, flag = estimate(u_key, prior)
    assert(flag == FLAG.NEGATIVE_PRIOR_DEPTH)

    u_key = np.array([110, 400])
    prior = Hypothesis(15.0, 0.2)
    hypothesis, flag = estimate(u_key, prior)
    assert(flag == FLAG.HYPOTHESIS_OUT_OF_SERCH_RANGE)

    u_key = np.array([390, 100])
    prior = Hypothesis(0.5, 0.2)
    hypothesis, flag = estimate(u_key, prior)
    assert(flag == FLAG.INSUFFICIENT_GRADIENT)

    # u_key is on the image edge
    u_key = np.array([0, 200])
    prior = Hypothesis(0.5, 0.2)
    hypothesis, flag = estimate(u_key, prior)
    assert(flag == FLAG.KEY_OUT_OF_RANGE)

    # very short search range
    u_key = np.array([110, 400])
    prior = Hypothesis(0.5, 0.001)
    hypothesis, flag = estimate(u_key, prior)
    assert(flag == FLAG.REF_EPIPOLAR_TOO_SHORT)

    u_key = np.array([110, 400])
    prior = Hypothesis(1 / keyframe.depth_map[u_key[1], u_key[0]], 0.01)
    hypothesis, flag = estimate(u_key, prior)
    assert(flag == FLAG.REF_OUT_OF_RANGE)

    x, y = u_key = np.array([420, 450])
    prior = Hypothesis(invert_depth(keyframe.depth_map[y, x]), 0.01)
    (inv_depth, variance), flag = estimate(u_key, prior)
    print("pred, true = ", invert_depth(inv_depth), keyframe.depth_map[y, x])
    assert(flag == FLAG.SUCCESS)
    assert(inv_depth > 0.0)
    assert(abs(invert_depth(inv_depth) - keyframe.depth_map[y, x]) < 1.0)
    assert(variance > 0.0)  # hard to test the value of variance
