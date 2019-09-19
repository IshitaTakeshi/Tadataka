from vitamine.pose_estimation import solve_pnp
from vitamine.so3 import rodrigues

class PoseManager(object):
    def __init__(self):
        self.rotations = []
        self.translations = []

    def add(self, R, t):
        self.rotations.append(R)
        self.translations.append(t)

    def get(self, i):
        R = self.rotations[i]
        t = self.translations[i]
        return R, t

    def get_motion_matrix(self, i):
        R, t = self.get(i)
        return motion_matrix(R, t)


class PoseEstimator(object):
    def __init__(self, matcher, points0, descriptors0a, descriptors0b):
        self.matcher = matcher
        self.points0 = points0
        self.descriptors0a = descriptors0a
        self.descriptors0b = descriptors0b

    def match_existing(self, keypoints1, descriptors1):
        """
        Match with descriptors that already have corresponding 3D points
        """
        matches01a = self.matcher(self.descriptors0a, descriptors1)
        matches01b = self.matcher(self.descriptors0b, descriptors1)

        if len(matches01a) > len(matches01b):
            return matches01a[:, 0], matches01a[:, 1]
        else:
            return matches01b[:, 0], matches01b[:, 1]

    def estimate(self, keypoints1, descriptors1):
        indices0, indices1 = self.match_existing(keypoints1, descriptors1)
        omega1, t1 = solve_pnp(self.points0[indices0], keypoints1[indices1])
        R1 = rodrigues(omega1.reshape(1, -1))[0]
        return R1, t1
