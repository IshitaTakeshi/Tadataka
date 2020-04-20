import numpy as np
from numpy.testing import assert_array_equal
from tadataka.vo.semi_dense._intensities import search_intensities


def test_intensity_search():
    intensities_key = np.array([1, -1, 2], dtype=np.float64)
    intensities_ref = np.array([-4, 3, 2, 4, -1, 3, 1], dtype=np.float64)
    # errors = [-4-3+4, 3-2+8, 2-4-2, 4+1+6, -1-3+2]
    #        = [-3, 9, -4, 11, 4]
    # argmin(errors) == 3
    # offset == 1 ( == len(intensities_key) // 2)
    # expected = argmin(errors) + offset

    index = search_intensities(intensities_key, intensities_ref)
    assert(index == 4)
