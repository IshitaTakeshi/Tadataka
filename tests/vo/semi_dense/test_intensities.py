import numpy as np
from numpy.testing import assert_array_equal
from tadataka.vo.semi_dense._intensities import search_intensities


def test_intensity_search():
    intensities_key = np.array([1, -1, 2], dtype=np.float64)
    intensities_ref = np.array([-4, 3, 2, 4, -1, 3, 1], dtype=np.float64)
    # errors = [25 + 16 + 0, 4 + 9 + 4, 1 + 25 + 9, 9 + 0 + 1, 4 + 16 + 1]
    #        = [41, 17, 35, 10, 21]
    # argmin(errors) == 3
    # offset == 1 ( == len(intensities_key) // 2)
    # expected = argmin(errors) + offset

    index = search_intensities(intensities_key, intensities_ref)
    assert(index == 4)
