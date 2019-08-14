from skimage import transform as tf
from skimage.measure import ransac

from vitamine.flow_estimation.keypoints import extract_keypoints, match
from vitamine.transform import AffineTransform


# TODO move to utils or somewhere
def affine_params_from_matrix(matrix):
    A, b = matrix[0:2, 0:2], matrix[0:2, 2]
    return A, b



class AffineTransformEstimator(object):
    def __init__(self):
        self.transformer = None

    def fit_keypoints(self, keypoints1, keypoints2):
        tform, inliers_mask = ransac((keypoints1, keypoints2),
                                     tf.AffineTransform,
                                     random_state=3939, min_samples=3,
                                     residual_threshold=2, max_trials=100)
        return affine_params_from_matrix(tform.params)

    def estimate(self, image1, image2):
        """
        Extract keypoints from each image and estimate affine correnpondence
        between them
        """
        keypoints1, descriptors1 = extract_keypoints(image1)
        keypoints2, descriptors2 = extract_keypoints(image2)
        matches12 = match(descriptors1, descriptors2)

        keypoints1 = keypoints1[matches12[:, 0]]
        keypoints2 = keypoints2[matches12[:, 1]]

        A, b = self.fit_keypoints(keypoints1, keypoints2)
        return AffineTransform(A, b)
