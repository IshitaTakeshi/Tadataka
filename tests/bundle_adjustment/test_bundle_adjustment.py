from autograd import numpy as np
from numpy.testing import assert_array_almost_equal

from vitamine.camera import CameraParameters
from vitamine.bundle_adjustment.parameters import ParameterConverter
from vitamine.bundle_adjustment.bundle_adjustment import (
    Transformer, MaskedResidual, BundleAdjustmentSolver, BundleAdjustment)


def test_transformer():
    converter = ParameterConverter()
    camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])

    omegas = np.array([
        [0, 0, np.pi / 2],
        [0, np.pi, 0],
    ])
    translations = np.array([
        [2, 1, 0],
        [0, 2, -1],
    ])
    points = np.array([
        [0, 1, 2],
        [-1, 0, 1],
        [-2, 1, 3],
        [3, 4, 1]
    ])

    params = converter.to_params(omegas, translations, points)

    transformer = Transformer(camera_parameters, converter)

    expected = np.array([
        [[1 / 2, 1 / 2],
         [2, 0],
         [1 / 3, -1 / 3],
         [-2, 4]],
        [[0, -1],
         [- 1 / 2, -1],
         [- 1 / 2, - 3 / 4],
         [3 / 2, -3]]
    ])
    assert_array_almost_equal(transformer.compute(params), expected)
