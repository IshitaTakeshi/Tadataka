from autograd import numpy as np

from vitamine.bundle_adjustment.bundle_adjustment import bundle_adjustment_core
from vitamine.bundle_adjustment.initializers import (
    PoseInitializer, PointInitializer)
from vitamine.bundle_adjustment.mask import correspondence_mask, compute_mask
from vitamine.bundle_adjustment.triangulation import (
    points_from_known_poses, MultipleTriangulation)
from vitamine.visual_odometry.local_ba import LocalBundleAdjustment
from vitamine.visual_odometry.extrema_tracker import (
    multiple_view_keypoints, TwoViewExtremaTracker)
from vitamine.visual_odometry.flow_estimation import estimate_affine_transform
from vitamine.bundle_adjustment.mask import pose_mask, point_mask
from vitamine.bundle_adjustment.pnp import estimate_pose
from vitamine.rigid.coordinates import camera_to_world
from vitamine.rigid.rotation import rodrigues
from vitamine.visual_odometry.initializers import Initializer
from vitamine.flow_estimation.image_curvature import compute_image_curvature
from vitamine.transform import AffineTransform
from vitamine.utils import is_in_image_range
from vitamine.map import Map


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


def init_affines(images):
    f = estimate_affine_transform
    return [f(images[i], images[i+1]) for i in range(0, len(images)-1)]


def init_curvatures(images):
    return [compute_image_curvature(image) for image in images]


class Window(object):
    def __init__(self, omegas, translations, curvatures, affines):
        self.omegas = omegas
        self.translations = translations
        self.curvatures = curvatures
        self.affines = affines

    def slide(self, new_omega, new_translation, new_curvature, new_affine):
        self.omegas = slide_window(self.omegas, new_omega)
        self.translations = slide_window(self.translations, new_translation)
        self.curvatures = slide_window(self.curvatures, new_curvature)
        self.affines = slide_window(self.affines, new_affine)
        return self


class NewPoseEstimation(object):
    def __init__(self, points, new_curvature, new_affine, K, lambda_):
        self.points = points
        self.tracker = TwoViewExtremaTracker(new_curvature, new_affine,
                                             lambda_)
        self.K = K

    def estimate(self, last_keypoints):
        new_keypoints = self.tracker.track(last_keypoints)
        new_omega, new_translation = estimate_pose(self.points, new_keypoints, self.K)
        return new_omega, new_translation


class NewPointTriangulation(object):
    def __init__(self, K):
        self.K = K

    def triangulate(self, omegas, translations, keypoints):
        t = MultipleTriangulation(omegas[:-1], translations[:-1],
                                  keypoints[:-1], self.K)
        return t.triangulate(omegas[-1], translations[-1], keypoints[-1])


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

    def initialize(self, images):
        curvatures = init_curvatures(images)
        affines = init_affines(images)

        viewpoint1, viewpoint2 = 0, 1

        keypoints = multiple_view_keypoints(curvatures, affines, self.lambda_)

        init = Initializer(keypoints, self.K)
        omegas, translations, points = init.initialize(viewpoint1, viewpoint2)

        window = Window(omegas, translations, curvatures, affines)

        return window, points, keypoints

    def sequence(self):
        global_map = Map()

        images = [self.observer.request() for i in range(self.window_size)]
        window, points, keypoints = self.initialize(images)

        global_map.add(
            *camera_to_world(window.omegas, window.translations),
            points
        )

        # omegas, translations, points = self.refine(
        #     omegas, translations, points, keypoint_matrix)

        last_image = images[-1]

        while self.observer.is_running():
            print("iter")

            last_keypoints = keypoints[-1]

            new_image = self.observer.request()

            new_affine = estimate_affine_transform(last_image, new_image)
            new_curvature = compute_image_curvature(new_image)

            estimator = NewPoseEstimation(points, new_curvature, new_affine,
                                          self.K, self.lambda_)
            new_omega, new_translation = estimator.estimate(last_keypoints)

            window.slide(new_omega, new_translation, new_curvature, new_affine)

            keypoints = multiple_view_keypoints(window.curvatures, window.affines,
                                                self.lambda_)

            triangulation = NewPointTriangulation(self.K)
            points = triangulation.triangulate(window.omegas, window.translations,
                                               keypoints)

            # omegas, translations, points = self.refine(omegas, translations, points, keypoints)
            last_image = new_image

            global_map.add(
                *camera_to_world(new_omega.reshape(1, -1),
                                 new_translation.reshape(1, -1)),
                points
            )

            # plot_map(global_map.camera_omegas,
            #          global_map.camera_locations,
            #          global_map.points)
