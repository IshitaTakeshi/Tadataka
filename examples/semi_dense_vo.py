import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

from tadataka.warp import warp2d, Warp2D
from tadataka.pose import Pose, estimate_pose_change
from tadataka.camera import CameraModel, CameraParameters
from tadataka.vo.semi_dense.age import increment_age
from tadataka import camera
from tadataka.feature import extract_features, Matcher
from tadataka.dataset import NewTsukubaDataset
from tadataka.vo.semi_dense.regularization import regularize
from tadataka.vo.semi_dense.hypothesis import HypothesisMap
from tadataka.vo.semi_dense.propagation import Propagation
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.vo.semi_dense.semi_dense import (
    InvDepthEstimator, InvDepthMapEstimator, ReferenceSelector
)

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


def make_estimator(camera_model, image):
    return InvDepthMapEstimator(
        InvDepthEstimator(camera_model, image,
                          inv_depth_range, **estimator_params)
    )


def to_relative(ref, pose_wk):
    camera_model_ref, image_ref, pose_wr = ref
    pose_rk = pose_wr.inv() * pose_wk
    return (camera_model_ref, image_ref, pose_rk.T)


def init_hypothesis(shape):
    inv_depth_map = np.random.uniform(*inv_depth_range, shape)
    variance_map = default_variance * np.ones(shape)
    return HypothesisMap(inv_depth_map, variance_map)


def update_hypothesis(estimator, reference_selector, prior_hypothesis):
    hypothesis, flag_map = estimator(prior_hypothesis, reference_selector)

    fused = fusion(prior_hypothesis, hypothesis)
    inv_depth_map = regularize(fused)

    return HypothesisMap(inv_depth_map, fused.variance_map), flag_map


def update(hypothesis0, age0, frame0, frame1, refframes0):
    camera_model0, image0, pose0 = frame0
    camera_model1, image1, pose1 = frame1
    warp10 = Warp2D(camera_model0, camera_model1, pose0, pose1)
    age1 = increment_age(age0, warp10, hypothesis0.depth_map)

    relative_frames = [to_relative(f, pose1) for f in refframes0]
    hypothesis1, flag_map1 = update_hypothesis(
        make_estimator(camera_model1, image1),
        ReferenceSelector(age1, relative_frames),
        propagate(warp10, hypothesis0)
    )
    return hypothesis1, age1, flag_map1


def plot(gt, hypothesis, age, flag_map):
    plot_depth(gt.image, age,
               flag_map, gt.depth_map,
               hypothesis.depth_map, hypothesis.variance_map)


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
    posew0 = Pose.identity()
    pose10 = estimate_initial_pose(gt0.camera_model, gt1.camera_model,
                                   gt0.image, gt1.image)

    t10_norm = 6.00
    pose10.t = t10_norm * pose10.t
    pose0w = posew0.inv()
    pose1w = pose10 * pose0w
    posew1 = pose1w.inv()

    print("pose10 pred", pose10)

    frame0 = camera_model0, image0, posew0
    frame1 = camera_model1, image1, posew1
    refframes0 = [frame0]
    hypothesis1, age1, flag_map1 = update(hypothesis0, age0,
                                          frame0, frame1, refframes0)
    plot(gt1, hypothesis1, age1, flag_map1)

    # frame2 = camera_model2, image2, pose2
    # refframes1 = [frame0, frame1]
    # hypothesis2, age2, flag_map2 = update(hypothesis1, age1,
    #                                       frame1, frame2, refframes1)
    # plot(gt1, hypothesis1, age1, flag_map1)

    # warp10 = Warp2D(camera_model0, camera_model1, pose0, pose1)
    # age1 = increment_age(age0, warp10, hypothesis0.depth_map)
    # hypothesis1 = update_hypothesis(
    #     propagate(warp10, hypothesis0),
    #     reference_selector(age1),
    #     key=(image1, pose1),
    #     refs=[(image0, pose0)]
    # )

    # pose21 = dvo(image1, image2, hypothesis1)
    # pose2 = pose21 * pose1
    # warp21 = Warp2D(camera_model1, camera_model2, pose1, pose2)
    # age2 = increment_age(age1, warp21, hypothesis1.depth_map)
    # hypothesis2 = update_hypothesis(
    #     propagate(warp21, hypothesis1),
    #     key=(image2, pose2),
    #     refs=[(image0, pose0),
    #           (image1, pose1)]
    # )

    # pose32 = dvo(image2, image3, hypothesis2)
    # pose3 = pose32 * pose2
    # warp32 = Warp3D(camera_model2, camera_model3, pose2, pose3)
    # age3 = increment_age(age2, warp32, hypothesis2.depth_map)
    # hypothesis3 = update_hypothesis(
    #     propagate(warp32, hypothesis2),
    #     key=(image3, pose3),
    #     refs=[(image0, pose0),
    #           (image1, pose1),
    #           (image2, pose2)]
    # )

    # pose43 = dvo(image3, image4, hypothesis3)
    # pose4 = pose43 * pose3
    # warp43 = Warp4D(camera_model3, camera_model4, pose3, pose4)
    # age4 = increment_age(age3, warp43, hypothesis3.depth_map)
    # hypothesis4 = update_hypothesis(
    #     propagate(warp43, hypothesis3),
    #     key=(image4, pose4),
    #     refs=[(image0, pose0),
    #           (image1, pose1),
    #           (image2, pose2),
    #           (image3, pose3)]
    # )



main()
