from autograd import numpy as np

from vitamine.bundle_adjustment.mask import (
    keypoint_mask, point_mask, fill_masked
)
from vitamine.bundle_adjustment.triangulation import two_view_reconstruction

from vitamine.optimization.initializers import BaseInitializer


def select_initial_viewpoints(keypoints):
    masks = keypoint_mask(keypoints)
    n_visible = np.sum(masks, axis=1)
    viewpoint1, viewpoint2 = np.argsort(n_visible)[::-1][0:2]
    mask = np.logical_and(masks[viewpoint1], masks[viewpoint2])
    return mask, viewpoint1, viewpoint2


class PointInitializer(object):
    def __init__(self, keypoints, K):
        self.keypoints = keypoints
        self.K = K

    def initialize(self):
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
    def __init__(self, keypoints, K):
        self.keypoints = keypoints
        self.K = K

    def initialize(self, points):
        # at least 4 corresponding points have to be found
        # between keypoitns and 3D poitns
        required_correspondences = 4

        # TODO make independent from cv2
        import cv2
        n_viewpoints = self.keypoints.shape[0]

        omegas = np.empty((n_viewpoints, 3))
        translations = np.empty((n_viewpoints, 3))

        masks = np.logical_and(
            point_mask(points),
            keypoint_mask(self.keypoints)
        )
        for i in range(n_viewpoints):
            if np.sum(masks[i]) < required_correspondences:
                omegas[i] = np.nan
                translations[i] = np.nan
                continue

            retval, rvec, tvec = cv2.solvePnP(
                points[masks[i]], self.keypoints[i, masks[i]],
                self.K, np.zeros(4)
            )
            omegas[i] = rvec.flatten()
            translations[i] = tvec.flatten()
        return omegas, translations


class Initializer(BaseInitializer):
    def __init__(self, keypoints, K):
        self.point_initializer = PointInitializer(keypoints, K)
        self.pose_initializer = PoseInitializer(keypoints, K)

    def initialize(self):
        """
        Initialize 3D points and camera poses

        keypoints : np.ndarray
            A set of keypoints of shape (n_viewpoints, 2)
        """

        points = self.point_initializer.initialize()
        omegas, translations = self.pose_initializer.initialize(points)
        return omegas, translations, points
