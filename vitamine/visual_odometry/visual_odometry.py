from autograd import numpy as np

from vitamine.bundle_adjustment.bundle_adjustment import bundle_adjustment_core
from vitamine.bundle_adjustment.initializers import (
    PoseInitializer, PointInitializer)
from vitamine.visual_odometry.local_ba import LocalBundleAdjustment
from vitamine.bundle_adjustment.mask import pose_mask, point_mask
from vitamine.bundle_adjustment.pnp import estimate_pose, estimate_poses
from vitamine.rigid.coordinates import camera_to_world
from vitamine.rigid.rotation import rodrigues
from vitamine.visual_odometry.initializers import Initializer, PointUpdater
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


class VisualOdometry(object):
    def __init__(self, observations, camera_parameters, window_size, start=0, end=None):
        n_observations = observations.shape[0]
        self.observations = observations
        self.camera_parameters = camera_parameters
        self.window_size = window_size
        self.start = start
        self.end = n_observations if end is None else max(n_observations, end)
        assert(self.start < self.end)

    def sequence(self):
        K = self.camera_parameters.matrix

        initial_omegas = None
        initial_translations = None
        initial_points = None

        for i in range(self.start, self.end-self.window_size+1):
            keypoints = self.observations[i:i+self.window_size]

            if initial_points is None:
                point_initializer = PointInitializer(keypoints, K)
                initial_points = point_initializer.initialize()

            pose_initializer = PoseInitializer(keypoints, K)
            initial_omegas, initial_translations =\
                pose_initializer.initialize(initial_points)

            mask = ParameterMask(initial_omegas, initial_translations,
                                 initial_points)
            params = to_params(*mask.get_masked())
            keypoints = mask.mask_keypoints(keypoints)

            params = bundle_adjustment_core(keypoints, params,
                                            mask.n_valid_viewpoints,
                                            mask.n_valid_points,
                                            self.camera_parameters)

            omegas, translations, points =\
                from_params(params, mask.n_valid_viewpoints, mask.n_valid_points)

            initial_omegas, initial_translations, initial_points =\
                mask.fill(omegas, translations, points)
            yield omegas, translations, points
