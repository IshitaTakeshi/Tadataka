from autograd import numpy as np
from numpy.testing import assert_array_equal

import skimage
from skimage import transform as tf
from skimage.color import rgb2gray
from skimage.feature import match_descriptors
from matplotlib import pyplot as plt

from vitamine.flow_estimation.keypoints import (
    MatchOneToMany, extract_keypoints, MatchMatrix)


def test_match_one_to_many():
    def ground_truth_keypoints():
        image0 = rgb2gray(skimage.data.astronaut())
        keypoints0, descriptors0 = extract_keypoints(image0)

        rs = range(0, 360, 90)
        keypoints = np.full((len(rs), keypoints0.shape[0], 2), np.nan)

        for i, r in enumerate(rs):
            image1 = tf.rotate(image0, r)
            keypoints1, descriptors1 = extract_keypoints(image1)
            matches01 = match_descriptors(descriptors0, descriptors1,
                                          metric="hamming", cross_check=False)
            assert(matches01.shape[0] == keypoints0.shape[0])

            keypoints[i, matches01[:, 0]] = keypoints1[matches01[:, 1]]
        return keypoints

    image0 = rgb2gray(skimage.data.astronaut())
    images = np.array([tf.rotate(image0, r) for r in range(0, 360, 90)])

    matcher = MatchOneToMany(image0)

    expected = ground_truth_keypoints()
    assert_array_equal(matcher.compute(images), expected)


def test_match_matrix():
    match_matrix = MatchMatrix()

    match_matrix.add(0, 1,
                     np.array([[0, 0],
                               [1, 2]]))

    match_matrix.add(0, 2,
                     np.array([[1, 0],
                               [3, 1]]))

    match_matrix.add(0, 3,
                     np.array([[3, 0],
                               [2, 3]]))

    match_matrix.add(1, 2,
                     np.array([[1, 2]]))

    match_matrix.add(1, 3,
                     np.array([[2, 1],
                               [1, 2]]))

    match_matrix.add(2, 3,
                     np.array([[4, 3]]))

    print("matrix")
    print(match_matrix.matrix())

    assert_array_equal(
        match_matrix.matrix(),
        np.array([
            [0, 0, np.nan, np.nan],
            [1, 2, 0, 1],
            [3, np.nan, 1, 0],
            [2, np.nan, 4, 3],
            [np.nan, 1, 2, 2]
        ])
    )
