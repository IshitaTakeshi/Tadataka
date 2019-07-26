from autograd import numpy as np
from vitamine.bundle_adjustment.triangulation import two_view_reconstruction
from vitamine.bundle_adjustment.mask import fill_masked
from vitamine.bundle_adjustment.pnp import estimate_poses

from vitamine.bundle_adjustment.mask import keypoint_mask
from vitamine.optimization.initializers import BaseInitializer


class PointInitializer(object):
    def __init__(self, keypoints, K, viewpoint1, viewpoint2):
        self.keypoints = keypoints
        self.K = K
        self.viewpoint1 = viewpoint1
        self.viewpoint2 = viewpoint2

    def initialize(self):
        masks = keypoint_mask(self.keypoints)
        mask = np.logical_and(masks[self.viewpoint1], masks[self.viewpoint2])

        R, t, points_ = two_view_reconstruction(
            self.keypoints[self.viewpoint1, mask],
            self.keypoints[self.viewpoint2, mask],
            self.K
        )

        points = fill_masked(points_, mask)
        return points


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
