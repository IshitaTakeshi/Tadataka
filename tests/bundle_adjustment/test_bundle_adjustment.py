from autograd import numpy as np
from numpy.testing import assert_array_almost_equal

from vitamine.camera import CameraParameters
from vitamine.bundle_adjustment.parameters import ParameterConverter
from vitamine.bundle_adjustment.bundle_adjustment import (
    Transformer, MaskedResidual, BundleAdjustmentSolver, bundle_adjustment)


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
    converter = ParameterConverter()

    params = converter.to_params(omegas_true, translations_true, points_true)

    transformer = Transformer(camera_parameters, converter)
    assert_array_almost_equal(transformer.compute(params), keypoints_true)


def test_bundle_adjustment():
    # TODO
    # BA has the scale / rotation ambiguity
    pass
