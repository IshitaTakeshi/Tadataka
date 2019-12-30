import numpy as np
from numpy.testing import assert_array_equal
from tadataka.feature.patch import extract_patches


def test_extract_patches():
    image = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24]
    ])
    keypoints = np.array([
#        x  y
        [2, 2],
        [2, 3],
        [1, 3]
    ])

    patches = extract_patches(image, keypoints, patch_size=3)
    assert_array_equal(patches[0],
                       [[6, 7, 8],
                        [11, 12, 13],
                        [16, 17, 18]])
    assert_array_equal(patches[1],
                       [[11, 12, 13],
                        [16, 17, 18],
                        [21, 22, 23]])
    assert_array_equal(patches[2],
                       [[10, 11, 12],
                        [15, 16, 17],
                        [20, 21, 22]])
