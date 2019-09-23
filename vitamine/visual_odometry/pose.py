from vitamine.pose_estimation import solve_pnp
from vitamine.so3 import rodrigues


class Pose(object):
    def __init__(self, R, t):
        self.R, self.t = R, t


def estimate_pose(points, keypoints):
    omega, t = solve_pnp(points, keypoints)
    R = rodrigues(omega.reshape(1, -1))[0]
    return R, t
