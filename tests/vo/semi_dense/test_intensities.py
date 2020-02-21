import numpy as np
from numpy.testing import assert_array_equal
from tadataka.vo.semi_dense.intensities import search_intensities, convolve


def test_intensity_search():
    intensities_key = np.array([1, -1, 2])
    intensities_ref = np.array([-4, 3, 2, 4, -1, 3, 1])
    # errors = [-4-3+4, 3-2+8, 2-4-2, 4+1+6, -1-3+2]
    #        = [-3, 9, -4, 11, 4]
    # argmin(errors) == 3
    # offset == 1 ( == len(intensities_key) // 2)
    # expected = argmin(errors) + offset

    index = search_intensities(intensities_key, intensities_ref)
    assert(index == 4)


def test_convolve():
    def calc_error(a, b):
        return np.dot(a, b)

    A = np.array([5, 3, -4, 1, 0, 9, 6, -7])
    B = np.array([-1, 3, 1])
    errors = convolve(A, B, calc_error)
    assert_array_equal(errors, [0, -14, 7, 8, 33, 2])
