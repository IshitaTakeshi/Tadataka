import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

from tadataka.warp import warp2d, Warp2D
from tadataka.pose import Pose, estimate_pose_change
from tadataka.camera import CameraModel
from tadataka import camera
from tadataka.vo.dvo import PoseChangeEstimator
from tadataka.feature import extract_features, Matcher
from tadataka.dataset import NewTsukubaDataset
from tadataka.vo.semi_dense.regularization import regularize
from tadataka.vo.semi_dense.hypothesis import HypothesisMap
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.numeric import safe_invert
from tadataka.vo.semi_dense.semi_dense import (
    InvDepthEstimator, InvDepthMapEstimator
)
from tadataka.vo.semi_dense.reference import make_reference_selector

from rust_bindings.semi_dense import (
    increment_age, propagate, update_depth, Frame, Params
)
from rust_bindings.camera import CameraParameters
from examples.plot import plot_depth


depth_range = 10.0, 1000.0
default_depth = 100.0
default_variance = 100.0
uncertaintity_bias = 1.0

params = Params(
    *depth_range,
    geo_coeff=0.2,
    photo_coeff=0.1,
    ref_step_size=0.005,
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
    return estimator(image0, depth_map0, image1, weights)


def wrap_(c):
    [fx, fy] = c.focal_length
    [ox, oy] = c.offset
    return CameraParameters((fx, fy), (ox, oy))


def get(frame):
    camera_params = wrap_(frame.camera_model.camera_parameters)
    image = rgb2gray(frame.image)
    return camera_params, image


def estimate_initial_pose(camera_model0, camera_model1, image0, image1):
    match = Matcher()

    features0 = extract_features(image0)
    features1 = extract_features(image1)
    matches01 = match(features0, features1)
    keypoints0 = features0.keypoints[matches01[:, 0]]
    keypoints1 = features1.keypoints[matches01[:, 1]]
    keypoints0 = camera_model0.normalize(keypoints0)
    keypoints1 = camera_model1.normalize(keypoints1)
    return estimate_pose_change(keypoints0, keypoints1)


def init_age(shape):
    return np.zeros(shape, dtype=np.uint64)


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


def main():
    dataset = NewTsukubaDataset("datasets/NewTsukubaStereoDataset")

    gt0 = dataset[200][0]
    gt1 = dataset[205][0]
    gt2 = dataset[210][0]
    gt3 = dataset[215][0]
    gt4 = dataset[220][0]

    camera_params0, image0 = get(gt0)
    camera_params1, image1 = get(gt1)
    camera_params2, image2 = get(gt2)
    camera_params3, image3 = get(gt3)
    camera_params4, image4 = get(gt4)

    age0 = init_age(image0.shape)
    depth_map0 = np.random.uniform(*depth_range, image0.shape)
    variance_map0 = default_variance * np.ones(image0.shape)

    posew0 = gt0.pose
    print("posew0 true", posew0)
    pose10 = estimate_initial_pose(gt0.camera_model, gt1.camera_model,
                                   gt0.image, gt1.image)

    # set scale manually since it cannot be known in the inital pose
    t10_norm = 6.00
    pose10.t = t10_norm * pose10.t
    posew1 = calc_next_pose(posew0, pose10)

    print("posew1 true", gt1.pose)
    print("posew1 pred", posew1)

    frame0 = Frame(camera_params0, image0, posew0.T)
    frame1 = Frame(camera_params1, image1, posew1.T)
    refframes0 = [frame0]

    depth_map1, variance_map1, age1, flag_map1 = update(
        depth_map0, variance_map0, age0,
        frame0, frame1, refframes0
    )
    plot(gt1, depth_map1, variance_map1, age1, flag_map1)

    # ==================================================

    pose21 = dvo(camera_params1, camera_params2,
                 image1, image2, depth_map1, variance_map1)
    posew2 = calc_next_pose(posew1, pose21)

    print("posew2 true", gt2.pose)
    print("posew2 pred", posew2)

    depth_map2, variance_map2 = propagate(
        pose21.T, camera_params1, camera_params2,
        depth_map1, variance_map1,
        default_depth, default_variance, uncertaintity_bias
    )

    frame2 = Frame(camera_params2, image2, posew2.T)
    refframes1 = [frame0, frame1]
    depth_map2, variance_map2, age2, flag_map2 = update(
        depth_map1, variance_map1, age1,
        frame1, frame2, refframes1
    )
    plot(gt2, depth_map2, variance_map2, age2, flag_map2)

    # ==================================================

    pose32 = dvo(camera_params2, camera_params3,
                 image2, image3, depth_map2, variance_map2)
    posew3 = calc_next_pose(posew2, pose32)

    print("posew3 true", gt3.pose)
    print("posew3 pred", posew3)

    depth_map3, variance_map3 = propagate(
        pose32.T, camera_params2, camera_params3,
        depth_map2, variance_map2,
        default_depth, default_variance, uncertaintity_bias
    )

    frame3 = Frame(camera_params3, image3, posew3.T)
    refframes2 = [frame0, frame1, frame2]
    depth_map3, variance_map3, age3, flag_map3 = update(
        depth_map2, variance_map2, age2,
        frame2, frame3, refframes2
    )
    plot(gt3, depth_map3, variance_map3, age3, flag_map3)


main()
