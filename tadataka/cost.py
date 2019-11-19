import numpy as np
from skimage.transform import ProjectiveTransform

from tadataka.stat import ChiSquaredTest


EPSILON = 1e-6


def transfer12(tform, keypoints1, keypoints2):
    return tform(keypoints1) - keypoints2


def transfer21(tform, keypoints1, keypoints2):
    return keypoints1 - tform.inverse(keypoints2)


def symmetric_transfer_filter(keypoints1, keypoints2, p=0.95):
    tform = ProjectiveTransform()
    success = tform.estimate(keypoints1, keypoints2)
    if not success:
        print_error("Failed to estimate homography")
        return None

    tester = ChiSquaredTest(p)

    D12 = transfer12(tform, keypoints1, keypoints2)
    D21 = transfer21(tform, keypoints1, keypoints2)
    mask = np.logical_and(tester.test(D12), tester.test(D21))
    return mask
