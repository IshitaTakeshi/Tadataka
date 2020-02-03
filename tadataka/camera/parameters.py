import numpy as np


class CameraParameters(object):
    def __init__(self, focal_length, offset, image_shape=None):
        assert(len(focal_length) == 2)
        assert(len(offset) == 2)

        self.focal_length = np.array(list(focal_length))
        self.offset = np.array(list(offset))

        if image_shape is None:
            self.image_shape = None
        else:
            self.image_shape = np.array(list(image_shape))

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
        return (list(self.image_shape) +
                list(self.focal_length) +
                list(self.offset))

    @staticmethod
    def from_params(params):
        return CameraParameters(image_shape=params[0:2],
                                focal_length=params[2:4],
                                offset=params[4:6])

    def __eq__(self, another):
        C1 = np.array_equal(self.image_shape, another.image_shape)
        C2 = np.array_equal(self.focal_length, another.focal_length)
        C3 = np.array_equal(self.offset, another.offset)
        return C1 and C2 and C3
