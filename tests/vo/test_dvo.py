import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

from tadataka.pose import WorldPose, LocalPose
from tadataka.dataset.new_tsukuba import NewTsukubaDataset
from tadataka.warp import LocalWarp2D, Warp2D
from tadataka.metric import photometric_error
from tadataka.vo.dvo import _PoseChangeEstimator, PoseChangeEstimator
from tadataka.vo.dvo import camera_model_at, image_shape_at
from tests.dataset.path import new_tsukuba


def test_subestimator():
    level = 6
    dataset = NewTsukubaDataset(new_tsukuba)
    frame0, frame1 = dataset[0][0], dataset[4][1]

    camera_model0 = camera_model_at(level, frame0.camera_model)
    camera_model1 = camera_model_at(level, frame1.camera_model)

    shape = image_shape_at(level, frame0.depth_map.shape)
    D0 = resize(frame0.depth_map, shape)
    I0 = resize(rgb2gray(frame0.image), shape)
    I1 = resize(rgb2gray(frame1.image), shape)

    estimator = _PoseChangeEstimator(camera_model0, camera_model1, max_iter=10)

    pose10_prior = LocalPose.identity()
    warp10 = LocalWarp2D(camera_model0, camera_model1, pose10_prior)
    error_prior = photometric_error(warp10, I0, D0, I1)

    weight_map = np.ones(shape)
    pose10 = estimator(I0, D0, I1, pose10_prior, weight_map)

    warp10 = LocalWarp2D(camera_model0, camera_model1, pose10)
    error_pred = photometric_error(warp10, I0, D0, I1)
    assert(error_pred < error_prior * 0.1)


def test_pose_change_estimator():
    dataset = NewTsukubaDataset(new_tsukuba)
    frame0, frame1 = dataset[0][1], dataset[4][0]

    camera_model0 = frame0.camera_model
    camera_model1 = frame1.camera_model

    D0 = frame0.depth_map
    I0 = rgb2gray(frame0.image)
    I1 = rgb2gray(frame1.image)

    estimator = PoseChangeEstimator(camera_model0, camera_model1,
                                    n_coarse_to_fine=6, max_iter=10)

    pose10_prior = WorldPose.identity()
    warp10 = LocalWarp2D(camera_model0, camera_model1, pose10_prior)
    error_prior = photometric_error(warp10, I0, D0, I1)

    weight_map = np.ones(D0.shape)
    pose10 = estimator(I0, D0, I1, weight_map, pose10_prior)

    warp10 = LocalWarp2D(camera_model0, camera_model1, pose10.to_local())
    error_pred = photometric_error(warp10, I0, D0, I1)
    assert(error_pred < error_prior * 0.12)
