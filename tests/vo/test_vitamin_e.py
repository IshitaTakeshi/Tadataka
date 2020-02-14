import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from skimage.transform import AffineTransform, warp
import pytest

from tadataka.flow_estimation.flow_estimation import estimate_affine_transform
from tadataka.feature import extract_features, Matcher
from tadataka.vo.vitamin_e import (
    Tracker, init_keypoint_frame, get_array, track_,
    match_keypoints, match_keypoint_ids,
    match_multiple_keypoint_ids)


match = Matcher(enable_ransac=False, enable_homography_filter=False)


def generate_checkerboard(n_squares, pixels_per_square):
    M = np.zeros((n_squares, n_squares), dtype=np.uint8)
    M[0::2, 0::2] = 255
    M[1::2, 1::2] = 255

    M = np.repeat(M, pixels_per_square, axis=0)
    M = np.repeat(M, pixels_per_square, axis=1)
    return M


@pytest.mark.skip(reason="Cannot reproduce the method")
def test_track_():
    affine1 = AffineTransform(rotation=0.02, translation=[5, 4])
    flow01_true = AffineTransform(rotation=0.02, translation=[3, 2])

    image0 = warp(generate_checkerboard(5, 10), affine1.inverse)
    image1 = warp(image0, flow01_true.inverse)

    def case1():
        flow01_pred = flow01_true
        keypoints0 = get_array(init_keypoint_frame(image0))
        keypoints1_pred = track_(keypoints0, image1, flow01_pred, lambda_=1e6)
        keypoints1_true = np.round(flow01_true(keypoints0))
        assert_array_almost_equal(keypoints1_true, keypoints1_pred)

    def case2():
        flow01_pred = AffineTransform(
            scale=flow01_true.scale,
            rotation=flow01_true.rotation+0.01,
            shear=flow01_true.shear,
            translation=flow01_true.translation
        )

        # transform by the noisy affine and
        # correct the error by extrema tracking
        keypoints0 = get_array(init_keypoint_frame(image0))

        curvature0 = compute_image_curvature(image0)
        curvature1 = compute_image_curvature(image1)

        keypoints1_true = get_array(init_keypoint_frame(image1))
        keypoints1_pred = track_(keypoints0, image1, flow01_pred, lambda_)
        keypoints1_pred = track_(keypoints0, image1, flow01_true, lambda_)

        assert_array_almost_equal(keypoints1_true, keypoints1_pred)

    def case3():
        features0 = extract_features(image0)
        features1 = extract_features(image1)

        matches01 = match(features0, features1)
        flow01_pred = estimate_affine_transform(
            features0.keypoints[matches01[:, 0]],
            features1.keypoints[matches01[:, 1]]
        )
        keypoints0 = get_array(init_keypoint_frame(image0))
        keypoints1_pred = track_(keypoints0, image1, flow01_pred, lambda_=1e3)
        keypoints1_true = flow01_true(keypoints0)
        assert_array_almost_equal(keypoints1_true, keypoints1_pred)

    case1()
    case2()
    case3()


@pytest.mark.skip(reason="Cannot reproduce the method")
def test_keypoint_tracking():
    affine1 = AffineTransform(rotation=0.02, translation=[5, 4])
    flow01_true = AffineTransform(translation=[3, 2])

    image0 = warp(generate_checkerboard(10, 20), affine1.inverse)
    image1 = warp(image0, flow01_true.inverse)

    features0 = extract_features(image0)
    features1 = extract_features(image1)

    matches01 = match(features0, features1)
    flow01_pred = estimate_affine_transform(
        features0.keypoints[matches01[:, 0]],
        features1.keypoints[matches01[:, 1]]
    )

    keypoints0 = init_keypoint_frame(image0)

    tracker = Tracker(flow01_true, image1, lambda_=1e8)
    keypoints1_pred = tracker(keypoints0)

    keypoints1_true = init_keypoint_frame(image1)

    matches = match_keypoints(keypoints1_true, keypoints1_pred)
    assert_array_almost_equal(
        get_array(keypoints1_true)[matches[:, 0]],
        get_array(keypoints1_pred)[matches[:, 1]]
    )


def test_match_keypoint_ids():
    ids0 = np.array([4, 5, 3, 2, 1])
    ids1 = np.array([1, 4, 6, 3, 0])
    matches01 = match_keypoint_ids(ids0, ids1)
    assert_array_equal(
        matches01,
        [[4, 0],   # 1
         [2, 3],   # 3
         [0, 1]]   # 4
    )


def test_match_multiple_keypoint_ids():
    #                0  1  2  3  4  5  6
    ids0 = np.array([4, 5, 3, 2, 1, 6])
    ids1 = np.array([1, 4, 6, 3, 5, 7, 8])
    ids2 = np.array([2, 3, 1, 0, 9, 6, 5, 4])
    ids3 = np.array([2, 6, 5, 3, 1, 9])
    matches = match_multiple_keypoint_ids([ids0, ids1, ids2, ids3])
    assert_array_equal(
        matches,
        [[4, 0, 2, 4],  # 1
         [2, 3, 1, 3],  # 3
         [1, 4, 6, 2],  # 5
         [5, 2, 5, 1]]  # 6
    )
