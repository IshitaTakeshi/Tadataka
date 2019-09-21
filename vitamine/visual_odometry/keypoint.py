from autograd import numpy as np


class PointIndices(object):
    def __init__(self, size):
        self.point_indices = -np.ones(size, dtype=np.int64)

    def subscribe(self, indices, point_indices):
        self.point_indices[indices] = point_indices

    @property
    def is_triangulated(self):
        return self.point_indices >= 0

    def get(self):
        return self.point_indices[self.is_triangulated]


class KeypointManager(object):
    def __init__(self):
        self.keypoints = []
        self.descriptors = []
        self.point_indices = []

    def add(self, keypoints, descriptors):
        self.keypoints.append(keypoints)
        self.descriptors.append(descriptors)
        self.point_indices.append(PointIndices(len(keypoints)))

    def add_triangulated(self, i, indices, point_indices):
        self.point_indices[i].subscribe(indices, point_indices)

    def get(self, i, indices_or_mask):
        keypoints = self.keypoints[i]
        descriptors = self.descriptors[i]
        return keypoints[indices_or_mask], descriptors[indices_or_mask]

    def get_triangulated(self, i):
        mask = self.point_indices[i].is_triangulated
        return self.get(i, mask)

    def get_untriangulated(self, i):
        mask = self.point_indices[i].is_triangulated
        return self.get(i, ~mask)

    def size(self, i):
        return len(self.keypoints[i])

    def get_point_indices(self, i):
        return self.point_indices[i].get()
