import numpy as np


class CameraParameters(object):
    def __init__(self, focal_length, offset):
        assert(len(focal_length) == 2)
        assert(len(offset) == 2)

        self.focal_length = np.array(list(focal_length))
        self.offset = np.array(list(offset))

    @property
    def matrix(self):
        ox, oy = self.offset
        fx, fy = self.focal_length

        return np.array([
            [fx, 0., ox],
            [0., fy, oy],
            [0., 0., 1.]
        ])

    @property
    def params(self):
        return list(self.focal_length) + list(self.offset)

    @staticmethod
    def from_params(params):
        return CameraParameters(focal_length=params[0:2],
                                offset=params[2:4])

    def __eq__(self, another):
        A = np.array_equal(self.focal_length, another.focal_length)
        B = np.array_equal(self.offset, another.offset)
        return A and B
