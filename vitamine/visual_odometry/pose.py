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


