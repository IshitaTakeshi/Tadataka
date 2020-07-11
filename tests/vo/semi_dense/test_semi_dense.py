import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_equal, assert_array_almost_equal)
from scipy.spatial.transform import Rotation
from skimage.color import rgb2gray

from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
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

from rust_bindings.semi_dense import Params, update_depth, Frame, estimate_debug_
from rust_bindings.camera import CameraParameters

from tests.dataset.path import new_tsukuba


def test_frame():
    camera_params = CameraParameters((10., 10.), (20., 10.))
    image = np.zeros((40, 20))
    transform = np.zeros((4, 4))
    frame = Frame(camera_params, image, transform)


def test_new_params():
    params = Params(
        min_depth=0.1,
        max_depth=10.0,
        geo_coeff=0.4,
        photo_coeff=0.5,
        ref_step_size=0.02,
        min_gradient=0.001
    )


def wrap_(c):
    [fx, fy] = c.focal_length
    [ox, oy] = c.offset
    return CameraParameters((fx, fy), (ox, oy))


def test_update_depth():
    dataset = NewTsukubaDataset(new_tsukuba)
    keyframe_, refframe_ = dataset[0]

    params = Params(
        min_depth=60.0,
        max_depth=1000.0,
        geo_coeff=0.01,
        photo_coeff=0.01,
        ref_step_size=0.01,
        min_gradient=0.2
    )

    key_camera_params = wrap_(keyframe_.camera_model.camera_parameters)
    ref_camera_params = wrap_(refframe_.camera_model.camera_parameters)
    T_wk = keyframe_.pose.T
    T_wr = refframe_.pose.T
    keyframe = Frame(key_camera_params, rgb2gray(keyframe_.image), T_wk)
    refframe = Frame(ref_camera_params, rgb2gray(refframe_.image), T_wr)

    shape = keyframe_.image.shape[0:2]
    key_image_ = rgb2gray(keyframe_.image)

    age_map = np.ones(shape, dtype=np.uint64)
    prior_depth = 200.0 * np.ones(shape, dtype=np.float64)
    prior_variance = np.ones(shape, dtype=np.float64)
    update_depth(
        keyframe,
        [refframe,],
        age_map,
        prior_depth,
        prior_variance,
        params,
    )


def test_estimate():
    dataset = NewTsukubaDataset(new_tsukuba)
    keyframe_, refframe_ = dataset[0]

    params = Params(
        min_depth=0.1,
        max_depth=1000.0,
        geo_coeff=0.01,
        photo_coeff=0.01,
        ref_step_size=0.01,
        min_gradient=0.2
    )

    key_camera_params = wrap_(keyframe_.camera_model.camera_parameters)
    ref_camera_params = wrap_(refframe_.camera_model.camera_parameters)
    T_wk = keyframe_.pose.T
    T_wr = refframe_.pose.T
    keyframe = Frame(key_camera_params, rgb2gray(keyframe_.image), T_wk)
    refframe = Frame(ref_camera_params, rgb2gray(refframe_.image), T_wr)

    pose_wk = keyframe_.pose
    pose_wr = refframe_.pose
    pose_rk = pose_wr.inv() * pose_wk

    def estimate(u_key, prior_depth, prior_variance):
        return estimate_debug_(u_key, prior_depth, prior_variance,
                               keyframe, refframe, params)

    u_key = np.array([110, 400])
    prior_depth = -10.0
    prior_variance = 10.0
    depth, variance, flag = estimate(u_key, prior_depth, prior_variance)
    assert(flag == FLAG.NEGATIVE_PRIOR_DEPTH)

    u_key = np.array([110, 400])
    prior_depth = 0.05
    prior_variance = 0.2
    depth, variance, flag = estimate(u_key, prior_depth, prior_variance)
    assert(flag == FLAG.HYPOTHESIS_OUT_OF_SERCH_RANGE)

    u_key = np.array([390, 100])
    prior_depth = 2.0
    prior_variance = 0.2
    depth, variance, flag = estimate(u_key, prior_depth, prior_variance)
    assert(flag == FLAG.INSUFFICIENT_GRADIENT)

    # u_key is on the image edge
    u_key = np.array([0, 200])
    prior_depth = 2.0
    prior_variance = 0.2
    depth, variance, flag = estimate(u_key, prior_depth, prior_variance)
    assert(flag == FLAG.KEY_OUT_OF_RANGE)

    # very short search range
    u_key = np.array([116, 400])
    prior_depth = 2.0
    prior_variance = 0.001
    depth, variance, flag = estimate(u_key, prior_depth, prior_variance)
    assert(flag == FLAG.REF_EPIPOLAR_TOO_SHORT)

    u_key = np.array([110, 400])
    prior_depth = keyframe_.depth_map[u_key[1], u_key[0]]
    prior_variance = 0.01
    depth, variance, flag = estimate(u_key, prior_depth, prior_variance)
    assert(flag == FLAG.REF_CLOSE_OUT_OF_RANGE)

    x, y = u_key = np.array([420, 450])
    prior_depth = keyframe_.depth_map[y, x]
    prior_variance = 0.01
    depth, variance, flag = estimate(u_key, prior_depth, prior_variance)
    assert(flag == FLAG.SUCCESS)
    assert(depth > 0.0)
    assert(abs(depth - keyframe_.depth_map[y, x]) < 1.0)
    assert(variance > 0.0)  # hard to test the exact value
