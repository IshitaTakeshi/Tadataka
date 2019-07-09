from autograd import numpy as np
from numpy.testing import assert_array_equal

from optimization import robustifiers


def test_squared_robustifier():
    robustifier = robustifiers.SquaredRobustifier()
    x = np.array([0.0, 1.0, 2.0])

    assert_array_equal(robustifier.robustify(x),
                       np.array([0.0, 1.0, 4.0]))

    # test methods of 'Robustifier' here
    assert_array_equal(robustifier.grad(x),
                       2 * x)

    assert_array_equal(robustifier.weights(x),
                       np.array([0.0, 2.0, 2.0]))


def test_geman_mcclure_robustifier():
    robustifier = robustifiers.GemanMcClureRobustifier(sigma=2.0)
    x = np.array([0.0, 1.0, 2.0])

    assert_array_equal(robustifier.robustify(x),
                       np.array([0.0, 0.2, 0.5]))
