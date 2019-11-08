# Copyright (C) 2011, the scikit-image team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  1. Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#  3. Neither the name of skimage nor the names of its contributors may be
#     used to endorse or promote products derived from this software without
#     specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import numpy as np
from skimage._shared.testing import assert_equal
from skimage import data
from skimage import transform as tf
from skimage.color import rgb2gray
from vitamine.match import match_binary_descriptors
from skimage.feature import BRIEF, corner_peaks, corner_harris
from skimage._shared import testing


def test_binary_descriptors_unequal_descriptor_sizes_error():
    """Sizes of descriptors of keypoints to be matched should be equal."""
    descs1 = np.array([[True, True, False, True],
                       [False, True, False, True]])
    descs2 = np.array([[True, False, False, True, False],
                       [False, True, True, True, False]])
    with testing.raises(ValueError):
        match_binary_descriptors(descs1, descs2)


def test_binary_descriptors():
    descs1 = np.array([[True, True, False, True, True],
                       [False, True, False, True, True]])
    descs2 = np.array([[True, False, False, True, False],
                       [False, False, True, True, True]])
    matches = match_binary_descriptors(descs1, descs2)
    assert_equal(matches, [[0, 0], [1, 1]])


def test_binary_descriptors_rotation_crosscheck_false():
    """Verify matched keypoints and their corresponding masks results between
    image and its rotated version with the expected keypoint pairs with
    cross_check disabled."""
    img = data.astronaut()
    img = rgb2gray(img)
    tform = tf.SimilarityTransform(scale=1, rotation=0.15, translation=(0, 0))
    rotated_img = tf.warp(img, tform, clip=False)

    extractor = BRIEF(descriptor_size=512)

    keypoints1 = corner_peaks(corner_harris(img), min_distance=5,
                              threshold_abs=0, threshold_rel=0.1)
    extractor.extract(img, keypoints1)
    descriptors1 = extractor.descriptors

    keypoints2 = corner_peaks(corner_harris(rotated_img), min_distance=5,
                              threshold_abs=0, threshold_rel=0.1)
    extractor.extract(rotated_img, keypoints2)
    descriptors2 = extractor.descriptors

    matches = match_binary_descriptors(descriptors1, descriptors2,
                                       cross_check=False)

    exp_matches1 = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                             24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46])
    exp_matches2 = np.array([ 0, 31,  2,  3,  1,  4,  6,  4, 38,  5, 27,  7,
                             13, 10,  9, 27,  7, 11, 15,  8, 23, 14, 12, 16,
                             10, 25, 18, 19, 21, 20, 41, 24, 25, 26, 28, 27,
                             22, 23, 29, 30, 31, 32, 35, 33, 34, 30, 36])
    assert_equal(matches[:, 0], exp_matches1)
    assert_equal(matches[:, 1], exp_matches2)


def test_binary_descriptors_rotation_crosscheck_true():
    """Verify matched keypoints and their corresponding masks results between
    image and its rotated version with the expected keypoint pairs with
    cross_check enabled."""
    img = data.astronaut()
    img = rgb2gray(img)
    tform = tf.SimilarityTransform(scale=1, rotation=0.15, translation=(0, 0))
    rotated_img = tf.warp(img, tform, clip=False)

    extractor = BRIEF(descriptor_size=512)

    keypoints1 = corner_peaks(corner_harris(img), min_distance=5,
                              threshold_abs=0, threshold_rel=0.1)
    extractor.extract(img, keypoints1)
    descriptors1 = extractor.descriptors

    keypoints2 = corner_peaks(corner_harris(rotated_img), min_distance=5,
                              threshold_abs=0, threshold_rel=0.1)
    extractor.extract(rotated_img, keypoints2)
    descriptors2 = extractor.descriptors

    matches = match_binary_descriptors(descriptors1, descriptors2,
                                       cross_check=True)

    exp_matches1 = np.array([ 0,  2,  3,  4,  5,  6,  9, 11, 12, 13, 14, 17,
                             18, 19, 21, 22, 23, 26, 27, 28, 29, 31, 32, 33,
                             34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46])
    exp_matches2 = np.array([ 0,  2,  3,  1,  4,  6,  5,  7, 13, 10,  9, 11,
                             15,  8, 14, 12, 16, 18, 19, 21, 20, 24, 25, 26,
                             28, 27, 22, 23, 29, 30, 31, 32, 35, 33, 34, 36])
    assert_equal(matches[:, 0], exp_matches1)
    assert_equal(matches[:, 1], exp_matches2)
