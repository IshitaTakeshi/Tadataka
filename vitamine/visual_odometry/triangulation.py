from vitamine.bundle_adjustment.triangulation import points_from_unknown_poses
from vitamine.coordinates import yx_to_xy, xy_to_yx
from vitamine.flow_estimation.keypoints import extract_keypoints, match
from vitamine.flow_estimation.extrema_tracker import ExtremaTracker
from vitamine.flow_estimation.flow_estimation import estimate_affine_transform
from vitamine.flow_estimation.image_curvature import compute_image_curvature
from vitamine.utils import round_int, is_in_image_range
from vitamine.transform import AffineTransform
from skimage.feature import peak_local_max

from skimage import transform as tf
from skimage.measure import ransac


def extract_local_maximums(curvature):
    local_maximums = peak_local_max(curvature, min_distance=20)
    return yx_to_xy(local_maximums)


def affine_from_matrix(matrix):
    A, b = matrix[0:2, 0:2], matrix[0:2, 2]
    return AffineTransform(A, b)


def triangulation_(curvature1, curvature2, affine_transform, K):
    image_shape = curvature1.shape[0:2]

    local_maximums1 = extract_local_maximums(curvature1)
    local_maximums2 = affine_transform.transform(local_maximums1)
    local_maximums2 = round_int(local_maximums2)

    mask = is_in_image_range(local_maximums2, image_shape)

    local_maximums1 = local_maximums1[mask]
    local_maximums2 = local_maximums2[mask]

    tracker = ExtremaTracker(curvature2, local_maximums2, lambda_=0.1)
    local_maximums2 = tracker.optimize()

    R, t, points = points_from_unknown_poses(local_maximums1,
                                             local_maximums2, K)
    assert(is_in_image_range(local_maximums2, image_shape).all())
    return points, local_maximums1, local_maximums2


def affine_inliers(keypoints1, keypoints2):
    tform, inliers_mask = ransac((keypoints1, keypoints2), tf.AffineTransform,
                                 random_state=3939, min_samples=3,
                                 residual_threshold=2, max_trials=100)
    return affine_from_matrix(tform.params), inliers_mask


def match_affine(image1, image2):
    """
    Extract keypoints from each image and estimate affine correnpondence
    between them
    """
    keypoints1, descriptors1 = extract_keypoints(image1)
    keypoints2, descriptors2 = extract_keypoints(image2)
    matches12 = match(descriptors1, descriptors2)

    keypoints1 = keypoints1[matches12[:, 0]]
    keypoints2 = keypoints2[matches12[:, 1]]

    affine_transform, inliers_mask = affine_inliers(keypoints1, keypoints2)

    return affine_transform, keypoints1[inliers_mask], keypoints2[inliers_mask]


def triangulation(image1, image2, K):
    curvature1 = compute_image_curvature(image1)
    curvature2 = compute_image_curvature(image2)

    affine_transform, keypoints1, keypoints2 = match_affine(image1, image2)

    return triangulation_(curvature1, curvature2, affine_transform, K)
