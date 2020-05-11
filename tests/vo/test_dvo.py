import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import pytest

from tadataka import camera
from tadataka.metric import PhotometricError
from tadataka.pose import Pose
from tadataka.dataset.new_tsukuba import NewTsukubaDataset
from tadataka.warp import LocalWarp2D, Warp2D
from tadataka.vo.dvo import _PoseChangeEstimator, PoseChangeEstimator
from tests.dataset.path import new_tsukuba


def get_resized(frame, scale):
    camera_model = camera.resize(frame.camera_model, scale)
    shape = (int(frame.image.shape[0] * scale),
             int(frame.image.shape[1] * scale))
    I = resize(rgb2gray(frame.image), shape)
    D = resize(frame.depth_map, shape)
    return camera_model, I, D


def test_pose_change_estimator():
    dataset = NewTsukubaDataset(new_tsukuba)

    frame0_, frame1_ = dataset[0][0], dataset[4][1]
    scale = 0.2
    camera_model0, I0, D0 = get_resized(frame0_, scale)
    camera_model1, I1, __ = get_resized(frame1_, scale)

    pose10_true = frame1_.pose.inv() * frame0_.pose

    error = PhotometricError(camera_model0, camera_model1, I0, D0, I1)

    estimator = PoseChangeEstimator(camera_model0, camera_model1,
                                    n_coarse_to_fine=5)

    def evaluate(weights, rate):
        pose_identity = Pose.identity()
        pose10_pred = estimator(I0, D0, I1, weights, pose_identity)
        assert(error(pose10_pred) < error(pose_identity))
        assert(error(pose10_pred) < error(pose10_true) * rate)

    print("before evaluate")
    evaluate(weights=None, rate=2.)
    evaluate(weights=np.ones(I0.shape), rate=2.)
    evaluate(weights="tukey", rate=3.)  # currently tukey cannot work well
    evaluate(weights="student-t", rate=2.)
    evaluate(weights="huber", rate=2.)

    with pytest.raises(ValueError, match="No such weights 'random'"):
        evaluate(weights="random", rate=2.)
