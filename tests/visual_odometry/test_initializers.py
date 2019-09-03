import pytest

from autograd import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal

from vitamine.visual_odometry.initializers import (
    initialize, select_new_viewpoint)
from vitamine.rigid.transformation import transform_all
from vitamine.rigid.rotation import rodrigues
from vitamine.camera import CameraParameters
from vitamine.projection.projections import PerspectiveProjection
from tests.utils import unit_uniform


def test_select_new_viewpoint():
    points = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [np.nan, np.nan, np.nan]
    ])

    keypoints = np.array([
        [[np.nan, np.nan],
         [np.nan, np.nan],
         [np.nan, np.nan]],
        [[1, 2],
         [3, 4],
         [np.nan, np.nan]],
        [[1, 2],
         [3, 4],
         [np.nan, np.nan]],
        [[1, 2],
         [np.nan, np.nan],
         [3, 4]],
    ])

    used_viewpoints = set([1])
    assert_equal(select_new_viewpoint(points, keypoints, used_viewpoints), 2)

    with pytest.raises(ValueError):
        select_new_viewpoint(points, keypoints,
                             used_viewpoints=set([0, 1, 2, 3]))



def transform_projection(projection, omegas, translations, points):
    P = transform_all(rodrigues(omegas), translations, points)
    shape = P.shape[0:2]
    P = projection.compute(P.reshape(-1, 3))
    return P.reshape(*shape, 2)


def test_initialize():
    camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])
    projection = PerspectiveProjection(camera_parameters)

    n_viewpoints = 5
    n_points = 9
    omegas = np.pi * unit_uniform((n_viewpoints, 3))
    translations = 10 * unit_uniform((n_viewpoints, 3))
    points = 10 * unit_uniform((n_points, 3))

    keypoints_true = transform_projection(
        projection,
        omegas, translations, points
    )
    omegas_pred, translations_pred, points_pred =\
        initialize(keypoints_true, 0, 1, camera_parameters.matrix)

    keypoints_pred = transform_projection(
        projection,
        omegas_pred, translations_pred, points_pred
    )

    assert_array_almost_equal(keypoints_true, keypoints_pred)
