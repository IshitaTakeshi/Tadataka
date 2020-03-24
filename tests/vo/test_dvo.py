import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

from tadataka.pose import WorldPose
from tadataka.dataset.new_tsukuba import NewTsukubaDataset
from tadataka.warp import Warp2D
from tadataka.metric import photometric_error
from tadataka.vo.dvo import _PoseChangeEstimator
from tadataka.vo.dvo import camera_model_at, image_shape_at
from tests.dataset.path import new_tsukuba


def test_calc_pose_update():
    level = 5
    dataset = NewTsukubaDataset(new_tsukuba)
    frame0, frame1 = dataset[0][0], dataset[4][1]

    camera_model0 = camera_model_at(level, frame0.camera_model)
    camera_model1 = camera_model_at(level, frame1.camera_model)

    shape = image_shape_at(level, frame0.depth_map.shape)
    D0 = resize(frame0.depth_map, shape)
    I0 = resize(rgb2gray(frame0.image), shape)
    I1 = resize(rgb2gray(frame1.image), shape)

    estimator = _PoseChangeEstimator(camera_model0, camera_model1, max_iter=10)

    prior = WorldPose.identity()
    warp = Warp2D(camera_model0, camera_model1, prior, WorldPose.identity())
    error_prior = photometric_error(warp, I0, D0, I1)

    weight_map = np.ones(shape)
    dpose = estimator(I0, D0, I1, prior, weight_map)

    warp = Warp2D(camera_model0, camera_model1, dpose, WorldPose.identity())
    error_pred = photometric_error(warp, I0, D0, I1)
    assert(error_pred < error_prior * 0.4)
