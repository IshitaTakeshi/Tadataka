from autograd import numpy as np
from vitamine.bundle_adjustment.triangulation import points_from_unknown_poses
from vitamine.bundle_adjustment.mask import fill_masked, correspondence_mask
from vitamine.bundle_adjustment.pnp import estimate_poses

from vitamine.bundle_adjustment.mask import keypoint_mask
from vitamine.optimization.initializers import BaseInitializer


class PointInitializer(object):
    def __init__(self, keypoints1, keypoints2, K):
        self.mask = correspondence_mask(keypoints1, keypoints2)
        self.keypoints1 = keypoints1[self.mask]
        self.keypoints2 = keypoints2[self.mask]
        self.K = K

    def initialize(self):
        R, t, points_ = points_from_unknown_poses(
            self.keypoints1,
            self.keypoints2,
            self.K
        )

        return fill_masked(points_, self.mask)


class PoseInitializer(object):
    def __init__(self, keypoints, K):
        self.keypoints = keypoints
        self.K = K

    def initialize(self, points):
        return estimate_poses(points, self.keypoints, self.K)


class Initializer(BaseInitializer):
    def __init__(self, keypoints, K,
                 initial_omegas, initial_translations,
                 initial_points):
        self.point_initializer = PointInitializer(
            keypoints, K, initial_points)
        self.pose_initializer = PoseInitializer(
            keypoints, K, initial_omegas, initial_translations)

    def initialize(self):
        """
        Initialize 3D points and camera poses

        keypoints : np.ndarray
            A set of keypoints of shape (n_viewpoints, 2)
        """
        points = self.point_initializer.initialize()
        omegas, translations = self.pose_initializer.initialize(points)
        return omegas, translations, points
