from autograd import numpy as np
import cv2
from skimage import img_as_ubyte

from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                             BRIEF, ORB)
from skimage import transform

from vitamine.coordinates import yx_to_xy, xy_to_yx


brief = BRIEF(
    descriptor_size=512,
    patch_size=64,
    mode="uniform",
    sigma=0.1
)

orb = ORB(n_keypoints=100)
star_detector = cv2.xfeatures2d.StarDetector_create()


def extract_brief(image):
    keypoints = star_detector.detect(img_as_ubyte(image), None)
    keypoints = np.array([list(p.pt) for p in keypoints])
    keypoints = xy_to_yx(keypoints)

    brief.extract(image, keypoints)
    keypoints = keypoints[brief.mask]
    descriptors = brief.descriptors
    return yx_to_xy(keypoints), descriptors


def extract_orb(image):
    orb.detect_and_extract(image)
    return yx_to_xy(orb.keypoints), orb.descriptors


extract_keypoints = extract_brief


def match(descriptors0, descriptors1):
    return match_descriptors(descriptors0, descriptors1,
                             metric="hamming", cross_check=True,
                             max_ratio=0.8)
