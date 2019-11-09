from collections import namedtuple

from autograd import numpy as np
import cv2
from skimage import img_as_ubyte

from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                             BRIEF, ORB)
from skimage import transform as tf
from skimage.measure import ransac

from vitamine.coordinates import yx_to_xy, xy_to_yx
from vitamine.cost import symmetric_transfer_filter
from vitamine.match import match_binary_descriptors



KeypointDescriptor = namedtuple("KeypointDescriptor",
                                ["keypoints", "descriptors"])


keypoint_detector = cv2.FastFeatureDetector_create(threshold=50)

brief = BRIEF(
    descriptor_size=512,
    patch_size=64,
    mode="uniform",
    sigma=0.1
)

orb = ORB(n_keypoints=100)


def extract_keypoints_(image):
    keypoints = keypoint_detector.detect(img_as_ubyte(image), None)
    if len(keypoints) == 0:
        return np.empty((0, 2), dtype=np.float64)
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


extract_keypoints =  extract_brief


empty_match = np.empty((0, 2), dtype=np.int64)


def match(descriptors0, descriptors1):
    return match_binary_descriptors(descriptors0, descriptors1,
                                    cross_check=True, max_ratio=0.8)


def ransac_affine(keypoints1, keypoints2):
    # estimate inliers using ransac on AffineTransform
    tform, inliers_mask = ransac((keypoints1, keypoints2),
                                 tf.AffineTransform,
                                 min_samples=2, random_state=3939,
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
    def __init__(self, enable_ransac=True, enable_homography_filter=True):
        self.enable_ransac = enable_ransac
        self.enable_homography_filter = enable_homography_filter

    def _ransac(self, keypoints1, keypoints2):
        assert(len(keypoints1) == len(keypoints2))
        _, inliers_mask = ransac_fundamental(keypoints1, keypoints2)
        return inliers_mask

    def __call__(self, kd1, kd2, min_inliers=12):
        # kd1, kd2 are instances of KeypointDescriptor
        keypoints1, descriptors1 = kd1
        keypoints2, descriptors2 = kd2

        if len(keypoints1) == 0 or len(keypoints2) == 0:
            return empty_match

        matches12 = match(descriptors1, descriptors2)

        if len(matches12) == 0:
            return empty_match

        if len(matches12) < min_inliers:
            return matches12

        if self.enable_ransac:
            _, mask = ransac_fundamental(keypoints1[matches12[:, 0]],
                                         keypoints2[matches12[:, 1]])
            matches12 = matches12[mask]

        if self.enable_homography_filter:
            mask = symmetric_transfer_filter(keypoints1[matches12[:, 0]],
                                             keypoints2[matches12[:, 1]],
                                             p=0.95)
            matches12 = matches12[mask]

        return matches12


def filter_matches(matches01, mask0, mask1):
    indices0, indices1 = matches01[:, 0], matches01[:, 1]
    mask = np.logical_and(mask0[indices0], mask1[indices1])
    return matches01[mask]
