from autograd import numpy as np

from vitamine.pose_estimation import estimate_pose as estimate
from vitamine.so3 import rodrigues


class Pose(object):
    def __init__(self, R, t):
        self.R, self.t = R, t

    @staticmethod
    def identity():
        return Pose(np.identity(3), np.zeros(3))

    def __eq__(self, other):
        return (np.isclose(self.R, other.R).all() and
                np.isclose(self.t, other.t).all())


def estimate_pose(points, point_indices0, keypoints1, matches01):
    indices0, indices1 = matches01[:, 0], matches01[:, 1]
    points_ = points.get(point_indices0[indices0])
    try:
        R1, t1 = estimate(points_, keypoints1[indices1])
    except NotEnoughInliersException:
        return None
    return Pose(R1, t1)

