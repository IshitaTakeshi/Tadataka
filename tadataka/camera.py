import numpy as np


class CameraParameters(object):
    def __init__(self, focal_length, offset, skew=0.):
        assert(len(focal_length) == 2)
        assert(len(offset) == 2)

        self.focal_length = focal_length
        self.offset = offset
        self.skew = skew

    @property
    def matrix(self):
        ox, oy = self.offset
        fx, fy = self.focal_length
        s = self.skew

        return np.array([
            [fx, s, ox],
            [0., fy, oy],
            [0., 0., 1.]
        ])
