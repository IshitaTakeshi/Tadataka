from autograd import numpy as np

from vitamine.bundle_adjustment.bundle_adjustment import bundle_adjustment_core
from vitamine.bundle_adjustment.initializers import (
    PoseInitializer, PointInitializer)
from vitamine.bundle_adjustment.mask import correspondence_mask
from vitamine.bundle_adjustment.triangulation import points_from_known_poses
from vitamine.visual_odometry.local_ba import LocalBundleAdjustment
from vitamine.bundle_adjustment.mask import pose_mask, point_mask
from vitamine.bundle_adjustment.pnp import estimate_pose, estimate_poses
from vitamine.rigid.coordinates import camera_to_world
from vitamine.rigid.rotation import rodrigues
from vitamine.visual_odometry.initializers import Initializer
from vitamine.assertion import check_keypoints
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


class PointUpdater(object):
    def __init__(self, omegas, translations, keypoints, K):
        self.omegas = omegas
        self.translations = translations
        self.keypoints = keypoints
        self.K = K

    def update(self, omega, translation, new_keypoints):
        R0 = rodrigues(omega.reshape(1, -1))[0]
        t0 = translation

        rotations = rodrigues(self.omegas)
        n_viewpoints, n_points = self.keypoints.shape[0:2]

        points = np.full((n_points, 3), np.nan)
        for i in range(n_viewpoints):
            R1, t1 = rotations[i], self.translations[i]
            keypoints_ = self.keypoints[i]

            mask = correspondence_mask(new_keypoints, keypoints_)

            # HACK Should we check 'depths_are_valid' ?
            points[mask], depths_are_valid = points_from_known_poses(
                R0, R1, t0, t1,
                new_keypoints[mask], keypoints_[mask], self.K
            )
        return points


class VisualOdometry(object):
    def __init__(self, observer, camera_parameters, window_size=8):
        self.observer = observer
        self.camera_parameters = camera_parameters
        self.window_size = window_size
        self.K = self.camera_parameters.matrix

    def initialize(self):
        keypoints = []
        for i in range(self.window_size):
            keypoints_ = self.observer.request()
            keypoints.append(keypoints_)
        keypoints = np.array(keypoints)

        # keypoints.shape == (window_size, n_points, 2)
        initializer = Initializer(keypoints, self.K)

        omegas, translations, points = initializer.initialize(0, 1)
        omegas, translations, points = self.refine(
            omegas, translations, points, keypoints)
        return omegas, translations, points, keypoints

    def refine(self, omegas, translations, points, keypoints):
        local_ba = LocalBundleAdjustment(omegas, translations, points,
                                         self.camera_parameters)
        omegas, translations, points = local_ba.compute(keypoints)
        return omegas, translations, points

    def sequence(self):
        global_map = Map()
        omegas, translations, points, keypoints = self.initialize()
        global_map.add(*camera_to_world(omegas, translations), points)

        np.set_printoptions(suppress=True)

        while self.observer.is_running():
            omegas_, translations_ = omegas[1:], translations[1:]
            keypoints_ = keypoints[1:]

            new_keypoints = self.observer.request()
            new_omega, new_translation = estimate_pose(points, new_keypoints,
                                                       self.K)

            updater = PointUpdater(omegas_, translations_, keypoints_, self.K)
            points = updater.update(new_omega, new_translation, new_keypoints)

            omegas[:-1] = omegas_
            omegas[-1] = new_omega
            translations[:-1] = translations_
            translations[-1] = new_translation
            keypoints[:-1] = keypoints_
            keypoints[-1] = new_keypoints

            omegas, translations, points = self.refine(
                omegas, translations, points, keypoints)

            global_map.add(*camera_to_world(omegas, translations), points)

