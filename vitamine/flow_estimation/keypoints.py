from autograd import numpy as np

from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                             BRIEF)
from skimage import transform

from vitamine.coordinates import yx_to_xy

extractor = BRIEF(mode="uniform")


def extract_keypoints(image):
    keypoints = corner_peaks(corner_harris(image), min_distance=5)
    extractor.extract(image, keypoints)
    keypoints = keypoints[extractor.mask]
    descriptors = extractor.descriptors
    return yx_to_xy(keypoints), descriptors


def match(descriptors0, descriptors1):
    return match_descriptors(descriptors0, descriptors1,
                             metric="hamming", cross_check=False)


from matplotlib import pyplot as plt
from skimage.feature import plot_matches


def plot(image0, image1, keypoints0, keypoints1, matches01):
    fig = plt.figure()

    ax = fig.add_subplot(211)
    ax.axis("off")
    ax.imshow(image0)

    ax = fig.add_subplot(212)
    ax.axis("off")
    ax.imshow(image1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_matches(ax, image0, image1, keypoints0, keypoints1, matches01)


class MatchOneToMany(object):
    def __init__(self, base_image):
        self.keypoints0, self.descriptors0 = extract_keypoints(base_image)
        self.base_image = base_image

    def compute(self, images):
        n_images = images.shape[0]
        n_keypoints = self.keypoints0.shape[0]

        keypoints = np.full((n_images, n_keypoints, 2), np.nan)
        for i, image in enumerate(images):
            keypoints1, descriptors1 = extract_keypoints(image)
            matches01 = match(self.descriptors0, descriptors1)
            keypoints[i, matches01[:, 0]] = keypoints1[matches01[:, 1]]
        return keypoints
