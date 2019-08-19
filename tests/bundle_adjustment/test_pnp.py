from autograd import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal)

from vitamine.bundle_adjustment.pnp import estimate_poses
from vitamine.camera import CameraParameters
from vitamine.rigid.transformation import transform_all
from vitamine.rigid.rotation import rodrigues
from vitamine.projection.projections import PerspectiveProjection

from tests.utils import project


def test_initialize():
    camera_parameters = CameraParameters(
        focal_length=[0.9, 1.2],
        offset=[0.8, -0.2]
    )

    rotations_true = np.array([
        [[0, 1, 0],
         [1, 0, 0],
         [0, 0, -1]],
        [[0, -1, 0],
         [0, 0, -1],
         [1, 0, 0]]
    ])

    translations_true = np.array([
        [0, 0, 3],
        [0, 0, 2]
    ])

    points = np.array([
        [0, 0, 0], [0, 0, 1],
        [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1],
        [1, 1, 0], [1, 1, 1],
    ], dtype=np.float64)

    projection = PerspectiveProjection(camera_parameters)
    keypoints_true = project(
        rotations_true, translations_true, points, projection)
    K = camera_parameters.matrix

    #------------------------------------------------------------------
    # the case if all 3D points are visible from all viewpoints
    #------------------------------------------------------------------
    omegas_pred, translations_pred = estimate_poses(points, keypoints_true, K)
    assert_array_almost_equal(rodrigues(omegas_pred), rotations_true)
    assert_array_almost_equal(translations_pred, translations_true)

    #-------------------------------------------------------------------
    # the case if only 4 points are visible from the 1st view point
    # although this is sufficient for estimating poses
    #-------------------------------------------------------------------

    keypoints = np.copy(keypoints_true)
    keypoints[1, 4:8] = np.nan
    omegas_pred, translations_pred = estimate_poses(points, keypoints, K)
    assert_array_almost_equal(rodrigues(omegas_pred), rotations_true)
    assert_array_almost_equal(translations_pred, translations_true)

    #------------------------------------------------------------------
    # the case if only 3 points are visible from the 1st view point
    #------------------------------------------------------------------
    keypoints = np.copy(keypoints_true)
    keypoints[1, 3:8] = np.nan
    omegas_pred, translations_pred = estimate_poses(points, keypoints, K)

    assert_equal(omegas_pred.shape, (2, 3))
    assert_equal(translations_pred.shape, (2, 3))

    # the 1st pose should be nan due to the lack of correspondences
    assert_array_equal(omegas_pred[1], np.nan)
    assert_array_equal(translations_pred[1], np.nan)

    # and the non-nan elements should be same as the ground truth
    assert_array_almost_equal(rodrigues(omegas_pred[[0]]), rotations_true[[0]])
    assert_array_almost_equal(translations_pred[0], translations_true[0])
