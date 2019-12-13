import warnings

import numpy as np
from tadataka import _triangulation as TR
from tadataka.pose import Pose
from tadataka.exceptions import InvalidDepthException
from tadataka.feature import empty_match


class TwoViewTriangulation(object):
    def __init__(self, pose0, pose1):
        self.triangulator = Triangulation([pose0, pose1])

    def triangulate(self, keypoints0, keypoints1):
        assert(keypoints0.shape == keypoints1.shape)
        keypoints = np.empty((keypoints0.shape[0], 2, 2))
        keypoints[:, 0] = keypoints0
        keypoints[:, 1] = keypoints1
        return self.triangulator.triangulate(keypoints)


class Triangulation(object):
    def __init__(self, poses, min_depth=0.0):
        self.n_poses = len(poses)
        self.rotations = np.array([pose.rotation.as_dcm() for pose in poses])
        self.translations = np.array([pose.t for pose in poses])
        self.min_depth = min_depth

    def triangulate(self, keypoints):
        """
        keypoints.shape == (n_keypoints, n_poses, 2)
        """
        return TR.linear_triangulation(self.rotations, self.translations,
                                       keypoints)
