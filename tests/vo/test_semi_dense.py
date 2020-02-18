from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.spatial.transform import Rotation

from tadataka.dataset.new_tsukuba import NewTsukubaDataset
from tadataka.pose import calc_relative_pose
from tadataka.vo.semi_dense import (
    convolve, coordinates_along_key_epipolar, coordinates_along_ref_epipolar,
    DepthEstimator, search_intensities
)


def test_convolve():
    def calc_error(a, b):
        return np.dot(a, b)

    A = np.array([5, 3, -4, 1, 0, 9, 6, -7])
    B = np.array([-1, 3, 1])
    errors = convolve(A, B, calc_error)
    assert_array_equal(errors, [0, -14, 7, 8, 33, 2])
    argmin = search_intensities(A, B, calc_error)
    assert(argmin == 1 + 1)  # + 1 for offest = (len(B) - 1) // 2


def test_coordinates_along_key_epipolar():
    t_key_to_ref = np.array([20, 30, 10])  # [2, 3]
    x_key = np.array([4, 1])
    # d = [1, -1]
    disparity = np.sqrt(2)
    assert_array_almost_equal(
        coordinates_along_key_epipolar(x_key, t_key_to_ref, disparity),
        [[4-2, 1+2],
         [4-1, 1+1],
         [4+0, 1-0],
         [4+1, 1-1],
         [4+2, 1-2]]
    )


def test_coordinates_along_ref_epipolar():
    inv_depths = np.array([0.001, 0.01, 0.1])
    x_key = np.array([2.0, 1.2])
    R = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    t = np.array([500, 200, 0])

    assert_array_almost_equal(
        coordinates_along_ref_epipolar(R, t, x_key, inv_depths),
        [[-0.75, -0.7],
         [-3.0, -1.6],
         [-25.5, -10.6]]
    )



# def test_depth_estimation():
#     dataset = NewTsukubaDataset(dataset_root)
#     keyframe, refframe = dataset[210]
#     pose_key_to_ref = calc_relative_pose(keyframe.pose, refframe.pose)
#     estimator = DepthEstimator(
#         keyframe.camera_model, refframe.camera_model,
#         keyframe.image, refframe.image,
#         pose_key_to_ref.world_to_local()
#     )
#
#     from matplotlib import pyplot as plt
#     plt.subplot(121)
#     plt.imshow(keyframe.image)
#     plt.subplot(122)
#     plt.imshow(refframe.image)
#
#     x, y = 450, 240
#     depth_pred = estimator(np.array([x, y]), 0.001, 1e1, 1.0)
#     print(keyframe.depth_map[y, x], depth_pred)
