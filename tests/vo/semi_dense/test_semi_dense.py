import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_equal, assert_array_almost_equal)
from scipy.spatial.transform import Rotation
from skimage.color import rgb2gray

from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tadataka.camera import CameraModel, CameraParameters
from tadataka.vo.semi_dense.hypothesis import Hypothesis
from tadataka.vo.semi_dense.frame import Frame
from tadataka.vo.semi_dense.semi_dense import (
    InvDepthEstimator, InvDepthMapEstimator
)
from tadataka.vo.semi_dense.reference import ReferenceSelector
from tadataka.vo.semi_dense.gradient import GradientImage
from tadataka.gradient import grad_x, grad_y
from tadataka.dataset import NewTsukubaDataset
from tadataka.coordinates import image_coordinates
from tadataka.numeric import safe_invert
from tadataka.vo.semi_dense.hypothesis import HypothesisMap

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
    assert(flag == FLAG.REF_CLOSE_OUT_OF_RANGE)

    # TODO what kind of input makes this condition true?
    # hypothesis, flag = estimate(u_key, prior)
    # assert(flag == FLAG.REF_FAR_OUT_OF_RANGE)

    x, y = u_key = np.array([420, 450])
    prior = Hypothesis(safe_invert(keyframe.depth_map[y, x]), 0.01)
    (inv_depth, variance), flag = estimate(u_key, prior)
    assert(flag == FLAG.SUCCESS)
    assert(inv_depth > 0.0)
    assert(abs(safe_invert(inv_depth) - keyframe.depth_map[y, x]) < 1.0)
    assert(variance > 0.0)  # hard to test the value of variance


def test_inv_depth_map_estimator():
    def estimator_(a, b, flag, u_key, prior):
        result = Hypothesis(prior.inv_depth * a, prior.variance * b)
        return result, flag

    prior_inv_depth_map = np.array([
        [10.0, 11.0],
        [12.0, 13.0]
    ])
    prior_variance_map = np.array([
        [100.0, 101.0],
        [102.0, 103.0]
    ])
    age_map = np.array([
        [1, 0],
        [3, 2]
    ])
    frames = [(2, 3, FLAG.NEGATIVE_PRIOR_DEPTH),
              (4, 5, FLAG.SUCCESS),
              (3, 0, FLAG.KEY_OUT_OF_RANGE)]

    selector = ReferenceSelector(age_map, frames)
    estimator = InvDepthMapEstimator(estimator_, selector)
    result, flag_map = estimator(
        HypothesisMap(prior_inv_depth_map, prior_variance_map)
    )

    assert_array_equal(result.inv_depth_map,
                       [[10.0 * 3, 11.0],
                        [12.0 * 2, 13.0 * 4]])
    assert_array_equal(result.variance_map,
                       [[100.0 * 0, 101.0],
                        [102.0 * 3, 103.0 * 5]])
    assert_array_equal(flag_map,
                       [[FLAG.KEY_OUT_OF_RANGE, FLAG.NOT_PROCESSED],
                        [FLAG.NEGATIVE_PRIOR_DEPTH, FLAG.SUCCESS]])
