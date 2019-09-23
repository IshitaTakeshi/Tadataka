from autograd import numpy as np
from vitamine.triangulation import points_from_known_poses


class Triangulation(object):
    def __init__(self, matcher, R1, t1, keypoints1, descriptors1):
        self.matcher = matcher
        self.R1 = R1
        self.t1 = t1
        self.keypoints1 = keypoints1
        self.descriptors1 = descriptors1

    def triangulate(self, R0, t0, keypoints0, descriptors0):
        matches01 = self.matcher(descriptors0, self.descriptors1)
        indices0, indices1 = matches01[:, 0], matches01[:, 1]

        points, valid_depth_mask = points_from_known_poses(
            R0, self.R1, t0, self.t1,
            keypoints0[indices0], self.keypoints1[indices1],
        )

        return points[valid_depth_mask], matches01[valid_depth_mask]
