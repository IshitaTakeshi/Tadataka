import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

from tadataka.warp import warp2d, Warp2D
from tadataka.camera import CameraModel
from tadataka.matrix import inv_motion_matrix
from tadataka.pose import Pose, estimate_pose_change
from tadataka import camera
from tadataka.vo.dvo import PoseChangeEstimator
from tadataka.feature import extract_features, Matcher
from tadataka.dataset import NewTsukubaDataset, TumRgbdDataset
from tadataka.vo.semi_dense.regularization import regularize
from tadataka.vo.semi_dense.hypothesis import HypothesisMap
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.camera.normalizer import Normalizer
from tadataka.numeric import safe_invert
from tadataka.vo.semi_dense.semi_dense import (
    InvDepthEstimator, InvDepthMapEstimator
)
from tadataka.vo.semi_dense.reference import make_reference_selector

from rust_bindings.semi_dense import (
    increment_age, propagate, update_depth, Frame, Params
)
from rust_bindings.camera import CameraParameters
from rust_bindings.semi_dense import estimate_debug_
from examples.plot import plot_depth


depth_range = 60.0, 1000.0 # depth_range = 0.1, 10.0
default_depth = 200.0  # default_depth = 1.0
default_variance = 100.0
uncertaintity_bias = 1.0

params = Params(
    *depth_range,
    geo_coeff=0.01,
    photo_coeff=0.01,
    ref_step_size=0.01,
    min_gradient=0.2
)


def dvo(camera_params0, camera_params1, image0, image1,
        depth_map0, variance_map0):
    estimator = PoseChangeEstimator(
        CameraModel(camera_params0, distortion_model=None),
        CameraModel(camera_params1, distortion_model=None),
        n_coarse_to_fine=7
    )
    weights = safe_invert(variance_map0)
    pose10 = estimator(image0, depth_map0, image1, weights)
    return pose10.T


def wrap_(c):
    [fx, fy] = c.focal_length
    [ox, oy] = c.offset
    return CameraParameters((fx, fy), (ox, oy))


def get(frame):
    camera_params = wrap_(frame.camera_model.camera_parameters)
    image = rgb2gray(frame.image)
    return camera_params, image


def estimate_initial_pose(camera_params0, camera_params1, image0, image1):
    match = Matcher()

    features0 = extract_features(image0)
    features1 = extract_features(image1)
    matches01 = match(features0, features1)
    keypoints0 = features0.keypoints[matches01[:, 0]]
    keypoints1 = features1.keypoints[matches01[:, 1]]
    keypoints0 = Normalizer(camera_params0).normalize(keypoints0)
    keypoints1 = Normalizer(camera_params1).normalize(keypoints1)
    return estimate_pose_change(keypoints0, keypoints1)


def update_hypothesis(estimator, prior_hypothesis):
    hypothesis, flag_map = estimator(prior_hypothesis)

    fused = fusion(prior_hypothesis, hypothesis)
    inv_depth_map = regularize(fused)

    return HypothesisMap(inv_depth_map, fused.variance_map), flag_map


def update(depth_map0, variance_map0, age0, frame0, frame1, refframes0):
    transform_w0 = frame0.transform
    transform_w1 = frame1.transform
    transform10 = np.linalg.inv(transform_w1).dot(transform_w0)

    age1 = increment_age(age0, frame0.camera_params, frame1.camera_params,
                         transform10, depth_map0)

    depth_map1, variance_map1, flag_map1 = update_depth(
        frame1, refframes0, age1,
        depth_map0, variance_map0, params
    )
    return depth_map1, variance_map1, age1, flag_map1


def plot(gt, depth_map, variance_map, age_map, flag_map):
    plot_depth(gt.image, age_map,
               flag_map, gt.depth_map,
               depth_map, variance_map)


def calc_next_pose(posew0, pose10):
    pose0w = posew0.inv()
    pose1w = pose10 * pose0w
    posew1 = pose1w.inv()
    return posew1


def align_scale(pose10, t10_norm):
    pose10.t = t10_norm * pose10.t
    return pose10


def init_pose10(camera_params0, camera_params1, image0, image1):
    pose10 = estimate_initial_pose(camera_params0, camera_params1, image0, image1)
    pose10 = align_scale(pose10, 6.00)
    return pose10.T


def calc_pose_w1(transform10, transform_w0):
    transform01 = inv_motion_matrix(transform10)
    transform_w1 = transform_w0.dot(transform01)
    return transform_w1


def calc_gt_pose10(posew0, posew1):
    pose10 = posew1.inv() * posew0
    return pose10.T


def estimate(keyframe, refframe,
             u_key, prior_depth, prior_variance):
    return estimate_debug_(u_key, prior_depth, prior_variance,
                           keyframe, refframe, params)


def make_frame(frame_):
    camera_params = wrap_(frame_.camera_model.camera_parameters)
    return Frame(camera_params, rgb2gray(frame_.image), frame_.pose.T)


def main():
    from tests.dataset.path import new_tsukuba
    dataset = NewTsukubaDataset("datasets/NewTsukubaStereoDataset")
    # dataset = NewTsukubaDataset(new_tsukuba)
    # dataset = TumRgbdDataset("datasets/rgbd_dataset_freiburg1_desk",
    #                          which_freiburg=1)

    K = 200
    N = 100

    gt0 = dataset[K][0]
    camera_params0, image0 = get(gt0)

    transform_w0 = gt0.pose.T
    frame0 = Frame(camera_params0, image0, transform_w0)

    refframes = [frame0]

    depth_map0 = init_depth_map(frame0.image.shape)
    variance_map0 = init_variance_map(frame0.image.shape)
    age0 = init_age(frame0.image.shape)

    for i in range(1, N):
        gt1 = dataset[K+i*5][0]
        camera_params1, image1 = get(gt1)

        if i == 1:
            transform10 = init_pose10(frame0.camera_params, camera_params1,
                                      frame0.image, image1)
        else:
            transform10 = dvo(frame0.camera_params, camera_params1,
                              frame0.image, image1, depth_map0, variance_map0)

        transform_w1 = calc_pose_w1(transform10, frame0.transform_wf)
        frame1 = Frame(camera_params1, image1, transform_w1)

        age1 = increment_age(age0, frame0.camera_params, frame1.camera_params,
                             transform10, depth_map0)

        depth_map1, variance_map1 = propagate(
            transform10, frame0.camera_params, frame1.camera_params,
            depth_map0, variance_map0,
            default_depth, default_variance, uncertaintity_bias
        )
        depth_map1, variance_map1, flag_map = update_depth(
            frame1, refframes, age1,
            depth_map1, variance_map1, params
        )

        plot(gt1, depth_map1, variance_map1, age1, flag_map)
        # TODO remove unused reference frames from 'refframes'
        refframes.append(frame1)

        depth_map0, variance_map0, age0 = depth_map1, variance_map1, age1
        frame0 = frame1
        gt0 = gt1


def init_age(shape):
    return np.zeros(shape, dtype=np.uint64)


def init_depth_map(shape):
    return np.random.uniform(*depth_range, shape)


def init_variance_map(shape):
    return default_variance * np.ones(shape)


main()
