from autograd import numpy as np
from numpy.testing import assert_array_equal

from vitamine.transform import AffineTransform
from vitamine.visual_odometry.extrema_tracker import (
    extract_local_maximums, extrema_tracking)
from tests.utils import set_hills


def array_to_set(coordinates):
    # convert to a set to ignore the order
    return set([tuple(p) for p in coordinates.tolist()])


def test_local_maximums():
    # set of coordinates
    expected = {
        (10, 20),
        (20, 10),
        (30, 40),
        (50, 30)
    }

    curvature = set_hills(np.zeros((100, 100)), expected)

    local_maximums = extract_local_maximums(curvature)

    assert(array_to_set(local_maximums) == expected)


def test_extrema_tracking():
    A = np.array([
        [0.8, 0.2],
        [-0.2, 0.8]
    ])
    b = np.array([2, 1])

    affine_transform = AffineTransform(A, b)

    # local_maximums2 = affine_transform(local_maximums1; A, b)
    # will be close to the expected coordinates 'local_maximums2_expected'
    # the residual to the expected coordinates will be corrected by
    # extrema tracking

    local_maximums1_expected = {
        (10, 50),
        (20, 65),
        (50, 35),
        (60, 50)
    }

    # local_maximums2 = affine_transform(local_maximums1; A, b) will be
    # [[20 39]
    #  [31 49]
    #  [49 19]
    #  [60 29]]

    local_maximums2_expected = {
        (20, 40),
        (30, 50),
        (50, 20),
        (60, 30)
    }

    curvature1 = set_hills(np.zeros((100, 100)), local_maximums1_expected)
    curvature2 = set_hills(np.zeros((100, 100)), local_maximums2_expected)

    local_maximums1, local_maximums2 =\
        extrema_tracking(curvature1, curvature2, affine_transform, lambda_=0.1)

    assert_array_equal(array_to_set(local_maximums1),
                       local_maximums1_expected)
    assert_array_equal(array_to_set(local_maximums2),
                       local_maximums2_expected)
