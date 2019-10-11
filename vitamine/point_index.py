from autograd import numpy as np


class PointIndices(object):
    def __init__(self, size):
        self.point_indices = -np.ones(size, dtype=np.int64)

    @property
    def is_triangulated(self):
        return self.point_indices >= 0

    @property
    def triangulated(self):
        return self.point_indices[self.is_triangulated]

    def subscribe(self, indices, point_indices):
        self.point_indices[indices] = point_indices

    def get(self, indices):
        assert(self.is_triangulated[indices].all())
        return self.point_indices[indices]

