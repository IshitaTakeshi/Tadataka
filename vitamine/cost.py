from autograd import numpy as np
from skimage.transform import ProjectiveTransform

from vitamine.stat import ChiSquaredTest


EPSILON = 1e-6


def transfer12(tform, keypoints1, keypoints2):
    return tform(keypoints1) - keypoints2


def transfer21(tform, keypoints1, keypoints2):
    return keypoints1 - tform.inverse(keypoints2)


def plot():
    from matplotlib import pyplot as plt

    fig = plt.figure()
    fig.suptitle("symmetric transfer differences")

    ax = fig.add_subplot(121)
    mask = ChiSquaredTest().test(P12)
    ax.scatter(X[mask, 0], X[mask, 1], label='inliers', color='b', marker='.')
    ax.scatter(X[~mask, 0], X[~mask, 1], label='outliers', color='r', marker='.')
    ax.grid(True)
    ax.legend()

    plt.show()


def transfer_outlier_detector(keypoints1, keypoints2):
    tform = ProjectiveTransform()
    success = tform.estimate(keypoints1, keypoints2)
    if not success:
        print_error("Failed to estimate homography")
        return None

    # e1 and e2 may follow the Chi-squared distribution
    D12 = transfer12(tform, keypoints1, keypoints2)
    D21 = transfer21(tform, keypoints1, keypoints2)
    tester = ChiSquaredTest()
    mask = np.logical_and(tester.test(D12), tester.test(D21))
    return mask
