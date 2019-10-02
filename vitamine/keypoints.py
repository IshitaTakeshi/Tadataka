from collections import namedtuple

from autograd import numpy as np
import cv2
from skimage import img_as_ubyte

from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                             BRIEF, ORB)
from skimage import transform as tf
from skimage.measure import ransac

from vitamine.coordinates import yx_to_xy, xy_to_yx


KeypointDescriptor = namedtuple("KeypointDescriptor",
                                ["keypoints", "descriptors"])


star_detector = cv2.xfeatures2d.StarDetector_create()

brief = BRIEF(
    descriptor_size=512,
    patch_size=64,
    mode="uniform",
    sigma=0.1
)

orb = ORB(n_keypoints=100)


def extract_keypoints_(image):
    keypoints = star_detector.detect(img_as_ubyte(image), None)
    return np.array([list(p.pt) for p in keypoints])


def extract_brief(image):
    keypoints = extract_keypoints_(image)
    keypoints = xy_to_yx(keypoints)

    brief.extract(image, keypoints)
    keypoints = keypoints[brief.mask]
    keypoints = yx_to_xy(keypoints)

    return KeypointDescriptor(keypoints, brief.descriptors)


def extract_orb(image):
    orb.detect_and_extract(image)
    keypoints = yx_to_xy(orb.keypoints)
    descriptors = orb.descriptors
    return KeypointDescriptor(keypoints, descriptors)


extract_keypoints = extract_brief


def match(descriptors0, descriptors1):
    return match_descriptors(descriptors0, descriptors1,
                             metric="hamming", cross_check=True,
                             max_ratio=0.8)


def ransac_affine(keypoints1, keypoints2):
    # estimate inliers using ransac on AffineTransform
    tform, inliers_mask = ransac((keypoints1, keypoints2),
                                 tf.AffineTransform,
                                 random_state=3939, min_samples=8,
                                 residual_threshold=1, max_trials=100)
    return tform.params, inliers_mask


def ransac_fundamental(keypoints1, keypoints2):
    # estimate inliers using ransac on FundamentalMatrixTransform
    tform, inliers_mask = ransac((keypoints1, keypoints2),
                                 tf.FundamentalMatrixTransform,
                                 random_state=3939, min_samples=8,
                                 residual_threshold=1, max_trials=100)
    return tform.params, inliers_mask


class Matcher(object):
    def __init__(self, enable_ransac=True):
        self.enable_ransac = enable_ransac

    def __call__(self, kd1, kd2):
        def empty_match():
            return np.empty((0, 2), dtype=np.int64)

        # kd1, kd2 are instances of KeypointDescriptor
        keypoints1, descriptors1 = kd1
        keypoints2, descriptors2 = kd2

        if len(keypoints1) == 0 or len(keypoints2) == 0:
            return empty_match()

        matches12 = match(descriptors1, descriptors2)

        if len(matches12) == 0:
            return empty_match()

        if not self.enable_ransac:
            return matches12

        keypoints1 = keypoints1[matches12[:, 0]]
        keypoints2 = keypoints2[matches12[:, 1]]

        if len(matches12) < 8:
            _, inliers_mask = ransac_affine(keypoints1, keypoints2)
            return matches12[inliers_mask]

        _, inliers_mask = ransac_fundamental(keypoints1, keypoints2)
        return matches12[inliers_mask]
