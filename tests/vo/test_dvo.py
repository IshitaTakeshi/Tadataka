import numpy as np
from numpy.testing import assert_array_equal
from skimage.color import rgb2gray
from scipy.spatial.transform import Rotation

from tadataka.dataset import NewTsukubaDataset
from tadataka.vo.dvo import PoseChangeEstimator
from tadataka.pose import Pose
from tadataka.matrix import motion_matrix
from tadataka.vo.dvo.projection import warp

from tests.dataset.path import new_tsukuba


def reprojection_error(dpose, camera_model, I0, D0, I1):
    G = motion_matrix(dpose.R, dpose.t)
    warped, mask = warp(camera_model.camera_parameters, I1, D0, G)
    assert(np.any(mask))
    return np.mean(np.power(warped[mask] - I0[mask], 2).flatten())


def test_pose_chage_estimator():
    dataset = NewTsukubaDataset(new_tsukuba)

    frame0 = dataset[0][0]
    frame1 = dataset[4][0]

    I0, D0 = rgb2gray(frame0.image), frame0.depth_map
    I1 = rgb2gray(frame1.image)
    camera_model = frame1.camera_model

    estimator = PoseChangeEstimator(camera_model, I0, D0, I1)
    dpose_pred = estimator.estimate()

    n_tests = 10
    error_pred = reprojection_error(
        dpose_pred, frame1.camera_model, I0, D0, I1
    )

    rotvec = dpose_pred.rotation.as_rotvec()
    for i in range(n_tests):
        noise = np.random.normal(-0.1, 0.1, size=rotvec.shape)
        dpose = Pose(Rotation.from_rotvec(rotvec + noise), dpose_pred.t)
        E = reprojection_error(dpose, camera_model, I0, D0, I1)
        assert(error_pred < E)

    t = dpose_pred.t
    for i in range(n_tests):
        noise = np.random.uniform(-3.0, 3.0, size=t.shape)
        dpose = Pose(dpose_pred.rotation, dpose_pred.t + noise)
        E = reprojection_error(dpose, camera_model, I0, D0, I1)
        assert(error_pred < E)
