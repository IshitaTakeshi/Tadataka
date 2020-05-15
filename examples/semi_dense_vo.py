import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

from tadataka.warp import warp2d, Warp2D
from tadataka.pose import Pose, estimate_pose_change
from tadataka.camera import CameraModel, CameraParameters
from tadataka import camera
from tadataka.vo.dvo import PoseChangeEstimator
from tadataka.feature import extract_features, Matcher
from tadataka.dataset import NewTsukubaDataset
from tadataka.vo.semi_dense.age import increment_age
from tadataka.vo.semi_dense.regularization import regularize
from tadataka.vo.semi_dense.hypothesis import HypothesisMap
from tadataka.vo.semi_dense.propagation import Propagation
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.numeric import safe_invert
from tadataka.vo.semi_dense.semi_dense import (
    InvDepthEstimator, InvDepthMapEstimator
)
from tadataka.vo.semi_dense.reference import make_reference_selector

from examples.plot import plot_depth


inv_depth_range = [1 / 1000, 1 / 60]
default_variance = 100

estimator_params = {
    "sigma_i": 0.1,
    "sigma_l": 0.2,
    "step_size_ref": 0.005,
    "min_gradient": 0.2
}

propagate = Propagation(default_inv_depth=1/200,
                        default_variance=default_variance,
                        uncertaintity_bias=1.0)

match = Matcher()


def dvo(camera_model0, camera_model1, image0, image1, hypothesis0):
    weights = safe_invert(hypothesis0.variance_map)
    estimator = PoseChangeEstimator(camera_model0, camera_model1,
                                    n_coarse_to_fine=7)
    return estimator(image0, hypothesis0.depth_map, image1, weights)


def to_perspective(camera_model):
    return CameraModel(camera_model.camera_parameters,
                       distortion_model=None)


def get(frame, scale=1.0):
    camera_model = to_perspective(frame.camera_model)
    camera_model = camera.resize(camera_model, scale)

    image = rgb2gray(frame.image)
    shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))
    image = resize(image, shape)
    return camera_model, image


def estimate_initial_pose(camera_model0, camera_model1, image0, image1):
    features0 = extract_features(image0)
    features1 = extract_features(image1)
    matches01 = match(features0, features1)
    keypoints0 = features0.keypoints[matches01[:, 0]]
    keypoints1 = features1.keypoints[matches01[:, 1]]
    keypoints0 = camera_model0.normalize(keypoints0)
    keypoints1 = camera_model1.normalize(keypoints1)
    return estimate_pose_change(keypoints0, keypoints1)


def init_age(shape):
    return np.zeros(shape, dtype=np.int64)


def make_estimator(frame1, age_map1, refframes0):
    camera_model, image, pose = frame1
    return InvDepthMapEstimator(
        InvDepthEstimator(camera_model, image,
                          inv_depth_range, **estimator_params),
        make_reference_selector(age_map1, refframes0, pose),
    )


def init_hypothesis(shape):
    inv_depth_map = np.random.uniform(*inv_depth_range, shape)
    variance_map = default_variance * np.ones(shape)
    return HypothesisMap(inv_depth_map, variance_map)


def update_hypothesis(estimator, prior_hypothesis):
    hypothesis, flag_map = estimator(prior_hypothesis)

    fused = fusion(prior_hypothesis, hypothesis)
    inv_depth_map = regularize(fused)

    return HypothesisMap(inv_depth_map, fused.variance_map), flag_map


def update(hypothesis0, age0, frame0, frame1, refframes0):
    camera_model0, _, pose0 = frame0
    camera_model1, _, pose1 = frame1
    warp10 = Warp2D(camera_model0, camera_model1, pose0, pose1)
    age1 = increment_age(age0, warp10, hypothesis0.depth_map)

    hypothesis1, flag_map1 = update_hypothesis(
        make_estimator(frame1, age1, refframes0),
        propagate(warp10, hypothesis0)
    )
    return hypothesis1, age1, flag_map1


def plot(gt, hypothesis, age, flag_map):
    plot_depth(gt.image, age,
               flag_map, gt.depth_map,
               hypothesis.depth_map, hypothesis.variance_map)


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
    camera_model0, image0 = get(gt0)
    camera_model1, image1 = get(gt1)
    camera_model2, image2 = get(gt2)
    camera_model3, image3 = get(gt3)
    camera_model4, image4 = get(gt4)

    age0 = init_age(image0.shape)
    hypothesis0 = init_hypothesis(image0.shape)

    posew0 = gt0.pose
    print("posew0", posew0)
    pose10 = estimate_initial_pose(gt0.camera_model, gt1.camera_model,
                                   gt0.image, gt1.image)

    t10_norm = 6.00
    pose10.t = t10_norm * pose10.t
    posew1 = calc_next_pose(posew0, pose10)

    print("posew1 true", gt1.pose)
    print("posew1 pred", posew1)

    frame0 = camera_model0, image0, posew0

    frame1 = camera_model1, image1, posew1
    refframes0 = [frame0]
    hypothesis1, age1, flag_map1 = update(hypothesis0, age0,
                                          frame0, frame1, refframes0)
    plot(gt1, hypothesis1, age1, flag_map1)

    # ==================================================

    pose21 = dvo(camera_model1, camera_model2,
                 image1, image2, hypothesis1)
    posew2 = calc_next_pose(posew1, pose21)

    print("posew2 true", gt2.pose)
    print("posew2 pred", posew2)

    frame2 = camera_model2, image2, posew2
    refframes1 = [frame0, frame1]
    hypothesis2, age2, flag_map2 = update(hypothesis1, age1,
                                          frame1, frame2, refframes1)
    plot(gt2, hypothesis2, age2, flag_map2)

    # ==================================================

    pose32 = dvo(camera_model2, camera_model3,
                 image2, image3, hypothesis2)
    posew3 = calc_next_pose(posew2, pose32)

    print("posew3 true", gt3.pose)
    print("posew3 pred", posew3)

    frame3 = camera_model3, image3, posew3
    refframes2 = [frame0, frame1, frame2]
    hypothesis3, age3, flag_map3 = update(hypothesis2, age2,
                                          frame2, frame3, refframes2)
    plot(gt3, hypothesis3, age3, flag_map3)

    # ==================================================

    pose43 = dvo(camera_model3, camera_model4,
                 image3, image4, hypothesis3)
    posew4 = calc_next_pose(posew3, pose43)

    print("posew4 true", gt4.pose)
    print("posew4 pred", posew4)

    frame4 = camera_model4, image4, posew4
    refframes3 = [frame0, frame1, frame2, frame3]
    hypothesis4, age4, flag_map4 = update(hypothesis3, age3,
                                          frame3, frame4, refframes3)
    plot(gt4, hypothesis4, age4, flag_map4)



main()
