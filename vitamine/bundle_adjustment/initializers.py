from autograd import numpy as np
from vitamine.bundle_adjustment.triangulation import two_view_reconstruction
from vitamine.bundle_adjustment.mask import fill_masked
from vitamine.bundle_adjustment.pnp import estimate_poses

from vitamine.bundle_adjustment.mask import keypoint_mask
from vitamine.optimization.initializers import BaseInitializer


def select_initial_viewpoints(keypoints):
    masks = keypoint_mask(keypoints)
    n_visible = np.sum(masks, axis=1)
    viewpoint1, viewpoint2 = np.argsort(n_visible)[::-1][0:2]
    mask = np.logical_and(masks[viewpoint1], masks[viewpoint2])
    return mask, viewpoint1, viewpoint2


class PointInitializer(object):
    def __init__(self, keypoints, K, initial_points=None):
        self.initial_points = initial_points

        self.keypoints = keypoints
        self.K = K

    def initialize(self):
        if self.initial_points is not None:
            return initial_points

        mask, viewpoint1, viewpoint2 = select_initial_viewpoints(
            self.keypoints
        )

        R, t, points_ = two_view_reconstruction(
            self.keypoints[viewpoint1, mask],
            self.keypoints[viewpoint2, mask],
            self.K
        )

        points = fill_masked(points_, mask)
        return points


class PoseInitializer(object):
    def __init__(self, keypoints, K,
                 initial_omegas=None, initial_translations=None):
        self.initial_omegas = initial_omegas
        self.initial_translations = initial_translations

        self.keypoints = keypoints
        self.K = K

    def initialize(self, points):
        initial_omegas = self.initial_omegas
        initial_translations = self.initial_translations

        if initial_omegas is not None and initial_translations is not None:
            return initial_omegas, initial_translations

        omegas, translations = estimate_poses(points, self.keypoints, self.K)

        if initial_omegas is not None:
            return initial_omegas, translations

        if initial_translations is not None:
            return omegas, initial_translations

        return omegas, translations


class Initializer(BaseInitializer):
    def __init__(self, keypoints, K,
                 initial_omegas=None, initial_translations=None,
                 initial_points=None):
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
