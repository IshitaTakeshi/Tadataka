from autograd import numpy as np

from vitamine.bundle_adjustment.bundle_adjustment import bundle_adjustment_core
from vitamine.bundle_adjustment.initializers import (
    PoseInitializer, PointInitializer)
from vitamine.bundle_adjustment.mask import correspondence_mask, compute_mask
from vitamine.bundle_adjustment.triangulation import (
    points_from_known_poses, MultipleTriangulation)
from vitamine.visual_odometry.local_ba import LocalBundleAdjustment
from vitamine.visual_odometry.flow_estimation import AffineTransformEstimator
from vitamine.visual_odometry.extrema_tracker import (
    multiple_view_keypoints, TwoViewExtremaTracker)
from vitamine.bundle_adjustment.mask import pose_mask, point_mask
from vitamine.bundle_adjustment.pnp import estimate_pose
from vitamine.rigid.coordinates import camera_to_world
from vitamine.rigid.rotation import rodrigues
from vitamine.visual_odometry.initializers import Initializer
from vitamine.flow_estimation.image_curvature import compute_image_curvature
from vitamine.assertion import check_keypoints
from vitamine.transform import AffineTransform
from vitamine.utils import is_in_image_range
from vitamine.map import Map


def count_redundancy(keypoints):
    # count the number of shared points like below
    # where the boolean matrix is the keypoint mask
    #
    #             point0  point1  point2  point3
    # viewpoint0 [  True   False    True   False]
    # viewpoint1 [  True    True   False   False]
    #   n shared       2       1       1       0
    # redundancy       2   +   1   +   1   +   0  =  4

    mask = keypoint_mask(keypoints)
    return np.sum(np.all(mask, axis=0))


def slide_window(array, new_element):
    array[:-1] = array[1:]
    array[-1] = new_element
    return array


def inverse_affine_params(A, b):
    N = A.shape[0]

    matrix = np.identity(N + 1)
    matrix[0:N, 0:N] = A
    matrix[0:N, N] = b
    inv_matrix = np.linalg.inv(matrix)
    A_inv = matrix[0:N, 0:N]
    b_inv = matrix[0:N, N]
    return A_inv, b_inv


def inverse(affine):
    A_inv, b_inv = inverse_affine_params(affine.A, affine.b)
    return AffineTransform(A_inv, b_inv)


def filter_unobserved(local_maximums, affine, image_shape):
    # keep local maximums extracted in the newly observed image area
    mask = is_in_image_range(inverse(affine).transform(local_maximums),
                             image_shape)
    return local_maximums[~mask]


def pool_images(observer, window_size):
    images = []
    for i in range(window_size):
        image = observer.request()
        images.append(image)
    return np.array(images)


def init_affines(images, affine_estimator):
    f = affine_estimator.estimate
    return [f(images[i], images[i+1]) for i in range(0, len(images)-1)]


def init_poses_and_points(keypoints, K):
    initializer = Initializer(keypoints, K)

    omegas, translations, points = initializer.initialize(0, 1)

    check_keypoints(keypoints, omegas, translations, points)

    return omegas, translations, points


def init_curvatures(images):
    return [compute_image_curvature(image) for image in images]


class NewPoseEstimator(object):
    def __init__(self, points, two_view_tracker, K, lambda_):
        # estimate the affine transform
        # from the latest image in the window to the new image

        # estimate the camera pose of the new (n-th) viewpoint
        self.points = points
        self.tracker = two_view_tracker

        self.K = K

    def estimate(self, last_keypoints):
        new_keypoints = self.tracker.track(last_keypoints)
        return estimate_pose(self.points, new_keypoints, self.K)


class VisualOdometry(object):
    def __init__(self, observer, camera_parameters, window_size=8):
        self.observer = observer
        self.camera_parameters = camera_parameters
        self.window_size = window_size
        self.K = self.camera_parameters.matrix
        self.lambda_ = 0.1

    def refine(self, omegas, translations, points, keypoints):
        local_ba = LocalBundleAdjustment(omegas, translations, points,
                                         self.camera_parameters)
        omegas, translations, points = local_ba.compute(keypoints)
        return omegas, translations, points

    def estimate_new_pose(self, points, last_keypoints, new_curvature, new_affine):
        tracker = TwoViewExtremaTracker(new_curvature, new_affine,
                                        self.lambda_)
        estimator = NewPoseEstimator(points, tracker,
                                     self.K, self.lambda_)
        return estimator.estimate(last_keypoints)

    def sequence(self):
        affine_estimator = AffineTransformEstimator()

        global_map = Map()

        images = pool_images(self.observer, self.window_size)
        curvatures = init_curvatures(images)
        affines = init_affines(images, affine_estimator)
        image_shape = images[0].shape[0:2]

        keypoints = multiple_view_keypoints(curvatures, affines, self.lambda_)

        omegas, translations, points = init_poses_and_points(keypoints, self.K)

        # omegas, translations, points = self.refine(
        #     omegas, translations, points, keypoint_matrix)


        while self.observer.is_running():
            new_image = self.observer.request()
            new_curvature = compute_image_curvature(new_image)
            new_affine = affine_estimator.estimate(images[-1], new_image)

            new_omega, new_translation = self.estimate_new_pose(
                points, keypoints[-1], new_curvature, new_affine)

            omegas = slide_window(omegas, new_omega)
            translations = slide_window(translations, new_translation)
            images = slide_window(images, new_image)
            curvatures = slide_window(curvatures, new_curvature)
            affines = slide_window(affines, new_affine)

            keypoints = multiple_view_keypoints(curvatures, affines,
                                                self.lambda_)

            triangulation = MultipleTriangulation(
                omegas[:-1],
                translations[:-1],
                keypoints[:-1],
                self.K
            )
            points = triangulation.triangulate(
                omegas[-1],
                translations[-1],
                keypoints[-1]
            )
            print(points)
            # omegas, translations, points = self.refine(omegas, translations, points, keypoints)

            # global_map.add(*camera_to_world(omegas, translations), added_points)
            # plot_map(global_map.camera_omegas,
            #          global_map.camera_locations,
            #          global_map.points)
