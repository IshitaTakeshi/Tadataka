import numpy as np

from tadataka.flow_estimation.regularizer import get_geman_mcclure


def test_geman_mcclure():
    geman_mcclure = get_geman_mcclure(1.)
    assert(geman_mcclure([1., 2.]) == 5. / 6.)

    geman_mcclure = get_geman_mcclure(2.)
    assert(geman_mcclure([1., 2.]) == 5. / 7.)
