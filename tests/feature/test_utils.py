import numpy as np
from numpy.testing import assert_array_equal
from tadataka.feature.utils import mask_border_keypoints


def test_mask_border_keypoints():
    image_shape = (400, 300)
    keypoints = np.array([
    #    x  y
        [0, 0],
        [1, 2],
        [2, 1],
        [2, 2],
        [297, 397],
        [298, 397],
        [297, 398],
        [299, 399],
    ])
    distance = 2

    assert_array_equal(
        mask_border_keypoints(image_shape, keypoints, distance),
        [False, False, False, True, True, False, False, False]
    )
