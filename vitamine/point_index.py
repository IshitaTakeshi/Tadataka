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

    def set_triangulated(self, indices, point_indices):
        if not np.all(self.point_indices[indices] == -1):
            raise ValueError(
                "You are trying to overwrite existing point indices"
            )
        if not np.all(point_indices >= 0):
            raise ValueError("Point indices cannot be negative")
        self.point_indices[indices] = point_indices

    @property
    def n_triangulated(self):
        return np.sum(self.is_triangulated)

    def get(self, indices):
        assert(self.is_triangulated[indices].all())
        return self.point_indices[indices]
