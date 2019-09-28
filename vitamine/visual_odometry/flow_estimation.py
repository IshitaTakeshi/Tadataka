from skimage import transform as tf
from skimage.measure import ransac

from vitamine.keypoints import extract_keypoints, match
from vitamine.coordinates import xy_to_yx
from vitamine.transform import AffineTransform


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


# TODO move to utils or somewhere
def affine_params_from_matrix(matrix):
    A, b = matrix[0:2, 0:2], matrix[0:2, 2]
    return A, b


def ransac_affine(keypoints1, keypoints2):
    # estimate inliers using ransac on AffineTransform
    tform, inliers_mask = ransac((keypoints1, keypoints2),
                                 tf.AffineTransform,
                                 random_state=3939, min_samples=8,
                                 residual_threshold=1, max_trials=5000)
    return tform.params, inliers_mask


def ransac_fundamental(keypoints1, keypoints2):
    # estimate inliers using ransac on FundamentalMatrixTransform
    tform, inliers_mask = ransac((keypoints1, keypoints2),
                                 tf.FundamentalMatrixTransform,
                                 random_state=3939, min_samples=8,
                                 residual_threshold=1, max_trials=5000)
    return tform.params, inliers_mask


def extract_and_match(image1, image2):
    keypoints1, descriptors1 = extract_keypoints(image1)
    keypoints2, descriptors2 = extract_keypoints(image2)
    matches12 = match(descriptors1, descriptors2)
    return keypoints1[matches12[:, 0]],  keypoints2[matches12[:, 1]]
