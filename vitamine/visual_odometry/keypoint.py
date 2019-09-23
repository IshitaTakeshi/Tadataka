from autograd import numpy as np


class Keypoints(object):
    def __init__(self, keypoints, descriptors):
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.point_indices = PointIndices(len(keypoints))

    def get(self, indices_or_mask=None):
        keypoints, descriptors = self.keypoints, self.descriptors
        if indices_or_mask is None:
            return keypoints, descriptors
        return keypoints[indices_or_mask], descriptors[indices_or_mask]

    def triangulated(self):
        """
        Get keypoints that have been used for triangulation.
        """
        mask = self.point_indices.is_triangulated
        return self.keypoints.get(mask)

    def untriangulated(self):
        """
        Get keypoints that have not been used for triangulation.
        These keypoints don't have corresponding 3D points.
        """
        mask = self.point_indices.is_triangulated
        return self.keypoints.get(~mask)

    def associate_points(self, indices, point_indices):
        self.point_indices.subscribe(indices, point_indices)


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
