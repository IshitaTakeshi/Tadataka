import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import pytest

from tadataka import camera
from tadataka.pose import WorldPose
from tadataka.dataset.new_tsukuba import NewTsukubaDataset
from tadataka.warp import Warp2D
from tadataka.metric import photometric_error
from tadataka.vo.dvo import _PoseChangeEstimator, PoseChangeEstimator
from tests.dataset.path import new_tsukuba


def load(scale):
    dataset = NewTsukubaDataset(new_tsukuba)
    frame0, frame1 = dataset[0][0], dataset[4][1]

    camera_model0 = camera.resize(frame0.camera_model, scale)
    camera_model1 = camera.resize(frame1.camera_model, scale)

    shape = (int(frame0.image.shape[0] * scale),
             int(frame0.image.shape[1] * scale))
    I0 = resize(rgb2gray(frame0.image), shape)
    D0 = resize(frame0.depth_map, shape)
    I1 = resize(rgb2gray(frame1.image), shape)
    return camera_model0, camera_model1, I0, D0, I1


def test_subestimator():
    camera_model0, camera_model1, I0, D0, I1 = load(1/10)

    estimator = _PoseChangeEstimator(camera_model0, camera_model1, max_iter=20)

    prior = WorldPose.identity()
    warp = Warp2D(camera_model0, camera_model1, prior, WorldPose.identity())
    error_prior = photometric_error(warp, I0, D0, I1)

    dpose = estimator(I0, D0, I1, prior, None)

    warp = Warp2D(camera_model0, camera_model1, dpose, WorldPose.identity())
    error_pred = photometric_error(warp, I0, D0, I1)
    assert(error_pred < error_prior * 0.12)


def test_pose_change_estimator():
    camera_model0, camera_model1, I0, D0, I1 = load(1/6)

    estimator = PoseChangeEstimator(camera_model0, camera_model1,
                                    n_coarse_to_fine=3)

    def error(pose10):
        # warp points in t0 coordinate onto the t1 coordinate
        # we regard pose1 as world origin
        warp = Warp2D(camera_model0, camera_model1,
                      pose10, WorldPose.identity())
        return photometric_error(warp, I0, D0, I1)

    def evaluate(weights, rate):
        identity = WorldPose.identity()
        pose10_pred = estimator(I0, D0, I1, weights, identity)
        assert(error(pose10_pred) < error(identity) * rate)

    evaluate(weights=None, rate=0.20)
    evaluate(weights=np.ones(I0.shape), rate=0.20)
    evaluate(weights="tukey", rate=0.40)  # currently tukey cannot work well
    evaluate(weights="student-t", rate=0.20)
    evaluate(weights="huber", rate=0.20)

    with pytest.raises(ValueError, match="No such weights 'random'"):
        evaluate(weights="random", rate=0.20)
