from skimage import transform as tf
from skimage.measure import ransac

from vitamine.flow_estimation.keypoints import extract_keypoints, match
from vitamine.coordinates import xy_to_yx
from vitamine.transform import AffineTransform


# TODO move to utils or somewhere
def affine_params_from_matrix(matrix):
    A, b = matrix[0:2, 0:2], matrix[0:2, 2]
    return A, b


def estimate_affine_from_keypoints(keypoints1, keypoints2):
    # estimate inliers using ransac on FundamentalMatrixTransform
    # it's possible to estimate AffineTransform in RANSAC, however,
    # we can get more inliers using FundamentalMatrixTransform
    tform, inliers_mask = ransac((keypoints1, keypoints2),
                                 tf.AffineTransform,
                                 random_state=3939, min_samples=8,
                                 residual_threshold=1, max_trials=5000)

    # estimate affine transform between two views using the estimated inliers
    A, b = affine_params_from_matrix(tform.params)
    return A, b, inliers_mask


# for debug
def plot_matches(image1, image2, keypoints1, keypoints2, inliers_mask):
    from matplotlib import pyplot as plt
    from skimage import feature
    from autograd import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title("number of inliers = {}".format(np.sum(inliers_mask)))
    indices = np.where(inliers_mask)[0]
    feature.plot_matches(ax, image1, image2,
                         xy_to_yx(keypoints1), xy_to_yx(keypoints2),
                         np.vstack((indices, indices)).T)
    plt.show()


def estimate_affine_transform(image1, image2):
    """
    Extract keypoints from each image and estimate affine correnpondence
    between them
    """

    keypoints1, descriptors1 = extract_keypoints(image1)
    keypoints2, descriptors2 = extract_keypoints(image2)

    matches12 = match(descriptors1, descriptors2)

    keypoints1 = keypoints1[matches12[:, 0]]
    keypoints2 = keypoints2[matches12[:, 1]]

    A, b, inliers_mask = estimate_affine_from_keypoints(keypoints1, keypoints2)
    # plot_matches(image1, image2, keypoints1, keypoints2, inliers_mask)
    return AffineTransform(A, b)
