from autograd import numpy as np
from numpy.testing import assert_array_almost_equal

from vitamine.bundle_adjustment.parameters import to_params, from_params
from vitamine.camera import CameraParameters
from vitamine.bundle_adjustment.bundle_adjustment import (
    Transformer, MaskedResidual, BundleAdjustmentSolver,
    bundle_adjustment_core)

from tests.utils import add_uniform_noise


camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])

omegas_true = np.array([
    [0, 0, np.pi / 2],
    [0, np.pi, 0],
])

translations_true = np.array([
    [2, 1, 0],
    [0, 2, 6],
])

# contains 9 points to test bundle adjustment
points_true = np.array([
    [0, 1, 2],
    [-1, 0, 1],
    [-2, 1, 3],
    [3, 4, 1],
    [1, 2, 5],
    [4, 3, 1],
    [-4, 2, 3],
    [-4, 2, 4],
    [3, 1, 2]
])

keypoints_true = np.array([
    [[1 / 2, 1 / 2],
     [2, 0],
     [1 / 3, -1 / 3],
     [-2, 4],
     [0, 2 / 5],
     [-1, 5],
     [0, -1],
     [0, -3 / 4],
     [1 / 2, 2]],
    [[0, 3 / 4],
     [1 / 5, 2 / 5],
     [2 / 3, 1],
     [-3 / 5, 6 / 5],
     [-1, 4],
     [-4 / 5, 1],
     [4 / 3, 4 / 3],
     [2, 2],
     [-3 / 4, 3 / 4]]
])


def test_transformer():
    params = to_params(omegas_true, translations_true, points_true)
    transformer = Transformer(camera_parameters, 2, 9)
    assert_array_almost_equal(transformer.compute(params), keypoints_true)


def test_bundle_adjustment():
    scale = 0.2
    omegas = add_uniform_noise(omegas_true, scale)
    translations = add_uniform_noise(translations_true, scale)
    points = add_uniform_noise(points_true, scale)

    params = to_params(omegas, translations, points)
    params = bundle_adjustment_core(keypoints_true, params, 2, 9,
                                    camera_parameters)
    omegas_pred, translations_pred, points_pred = from_params(params, 2, 9)

    assert_array_almost_equal(omegas_pred, omegas_true, decimal=1)
    assert_array_almost_equal(translations_pred, translations_true, decimal=1)
    assert_array_almost_equal(points_pred, points_true, decimal=1)
