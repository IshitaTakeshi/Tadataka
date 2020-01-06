import numpy as np
from numpy.testing import assert_array_equal

from tadataka.flow_estimation.regularizer import get_geman_mcclure


def test_geman_mcclure():
    X = np.array([
        [1, 2],
        [3, 4],
        [-1, -2],
        [0, 0]
    ])


    geman_mcclure = get_geman_mcclure(sigma=1)
    assert_array_equal(geman_mcclure(X), [5 / 6, 25 / 26, 5 / 6, 0])

    geman_mcclure = get_geman_mcclure(sigma=2)
    assert_array_equal(geman_mcclure(X), [5 / 9, 25 / 29, 5 / 9, 0])
