import numpy as np

from tadataka.flow_estimation.regularizer import GemanMcClure


def test_geman_mcclure():
    geman_mcclure = GemanMcClure(1.)
    assert(geman_mcclure.compute([1., 2.]) == 5. / 6.)

    geman_mcclure = GemanMcClure(2.)
    assert(geman_mcclure.compute([1., 2.]) == 5. / 9.)
