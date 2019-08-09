from autograd import numpy as np
from matplotlib import pyplot as plt

from vitamine.bundle_adjustment.mask import correspondence_mask
from vitamine.bundle_adjustment.triangulation import points_from_unknown_poses
from vitamine.coordinates import yx_to_xy
from vitamine.flow_estimation.extrema_tracker import ExtremaTracker
from vitamine.flow_estimation.image_curvature import compute_image_curvature
from vitamine.flow_estimation.flow_estimation import estimate_affine_transform
from vitamine.flow_estimation.keypoints import extract_keypoints
from vitamine.utils import round_int, is_in_image_range
from vitamine.matrix import affine_trasform
from vitamine.visualization.visualizers import plot3d

from skimage.feature import peak_local_max, plot_matches
from vitamine.flow_estimation.keypoints import match


class AffineTransformEstimation(object):
    def __init__(self, keypoint_matrix):
        self.keypoint_matrix = keypoint_matrix
        self.A = None
        self.b = None

    def fit(self, viewpoint1, viewpoint2):
        keypoints1 = self.keypoint_matrix[viewpoint1]
        keypoints2 = self.keypoint_matrix[viewpoint2]

        mask = correspondence_mask(keypoints1, keypoints2)

        self.A, self.b = estimate_affine_transform(keypoints1[mask],
                                                   keypoints2[mask])

    def transform(self, coordinates):
        coordinates = affine_trasform(coordinates, self.A, self.b)
        return round_int(coordinates)


def extract_local_maximums(curvature):
    local_maximums = peak_local_max(curvature, min_distance=20)
    return yx_to_xy(local_maximums)


def triangulation_(keypoints1, keypoints2, curvatures1, curvatures2,
                   camera_parameters):
    K = camera_parameters.matrix
    image_shape = curvatures1.shape[0:2]

    A, b = estimate_affine_transform(keypoints1, keypoints2)

    local_maximums1 = extract_local_maximums(curvatures1)
    local_maximums2 = round_int(affine_trasform(local_maximums1, A, b))

    mask = is_in_image_range(local_maximums2, image_shape)

    local_maximums1 = local_maximums1[mask]
    local_maximums2 = local_maximums2[mask]

    tracker = ExtremaTracker(curvatures2, local_maximums2,
                             lambda_=0.1)
    local_maximums2 = tracker.optimize()

    R, t, points = points_from_unknown_poses(local_maximums1,
                                             local_maximums2, K)

    assert(is_in_image_range(local_maximums2, image_shape).all())
    return points, local_maximums1, local_maximums2


def triangulation(image1, image2, camera_parameters):
    keypoints1, descriptors1 = extract_keypoints(image1)
    keypoints2, descriptors2 = extract_keypoints(image2)
    matches12 = match(descriptors1, descriptors2)

    curvature1 = compute_image_curvature(image1)
    curvature2 = compute_image_curvature(image2)

    points, local_maximums1, local_maximums2 = triangulation_(
            keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]],
            curvature1, curvature2, camera_parameters)

    assert(local_maximums1.shape == local_maximums2.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Keypoint matching")
    plot_matches(ax, image1, image2, keypoints1, keypoints2, matches12)

    N = local_maximums1.shape[0]
    matches12 = np.vstack((np.arange(N), np.arange(N))).T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Curvature extrema tracking")
    plot_matches(ax, image1, image2, local_maximums1, local_maximums2, matches12)

    plot3d(points)

    plt.show()
