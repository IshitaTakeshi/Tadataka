from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.spatial.transform import Rotation

from tadataka.dataset.new_tsukuba import NewTsukubaDataset
from tadataka.dataset.tum_rgbd import TumRgbdDataset
from tadataka.pose import calc_relative_pose
from tadataka.vo.semi_dense import (
    convolve,
    DepthEstimator, search_intensities
)

from tests.dataset.path import new_tsukuba


def test_convolve():
    def calc_error(a, b):
        return np.dot(a, b)

    A = np.array([5, 3, -4, 1, 0, 9, 6, -7])
    B = np.array([-1, 3, 1])
    errors = convolve(A, B, calc_error)
    assert_array_equal(errors, [0, -14, 7, 8, 33, 2])
    argmin = search_intensities(A, B, calc_error)
    assert(argmin == 1)


# dataset_root = Path("datasets/rgbd_dataset_freiburg1_desk")

from skimage.color import rgb2gray

def test_depth_estimation():
    dataset = NewTsukubaDataset(new_tsukuba)
    keyframe, refframe = dataset[0] # , dataset[108]

    pose_key_to_ref = calc_relative_pose(keyframe.pose, refframe.pose)
    estimator = DepthEstimator(
        keyframe.camera_model, refframe.camera_model,
        rgb2gray(keyframe.image), rgb2gray(refframe.image),
        pose_key_to_ref.world_to_local()
    )

    x, y = 490, 240
    depth_pred = estimator(np.array([x, y]), 0.001, 1e1, 1.0)
    # print(keyframe.depth_map[y, x], depth_pred)
