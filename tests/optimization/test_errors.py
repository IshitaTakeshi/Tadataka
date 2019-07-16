from autograd import numpy as np
from numpy.testing import assert_equal
from vitamine.optimization.robustifiers import SquaredRobustifier
from vitamine.optimization.errors import SumRobustifiedNormError


def test_sum_robustified_norm_error():
    robustifier = SquaredRobustifier()
    residuals = np.array([
        [4.0, 3.0],
        [-1.0, 2.0],
        [3.0, 0.0]
    ])
    expected = 25.0 + 5.0 + 9.0
    assert_equal(
        SumRobustifiedNormError(robustifier).compute(residuals),
        expected
    )
