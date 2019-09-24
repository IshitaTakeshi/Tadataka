from collections import namedtuple

from autograd import numpy as np


# just to enable accessing by name
# ex. localfeatures.triangulated.descriptors
KD = namedtuple("KeypointDescriptor", ["keypoints", "descriptors"])


class LocalFeatures(object):
    def __init__(self, keypoints, descriptors):
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.point_indices = -np.ones(len(keypoints), dtype=np.int64)

    def get(self, indices_or_mask=None):
        if indices_or_mask is None:
            return KD(self.keypoints, self.descriptors)
        return KD(self.keypoints[indices_or_mask],
                  self.descriptors[indices_or_mask])

    @property
    def is_triangulated(self):
        return self.point_indices >= 0

    @property
    def triangulated(self):
        """
        Get keypoints that have been used for triangulation.
        """
        mask = self.point_indices.is_triangulated
        return self.keypoints.get(mask)

    @property
    def untriangulated(self):
        """
        Get keypoints that have not been used for triangulation.
        These keypoints don't have corresponding 3D points.
        """
        mask = self.point_indices.is_triangulated
        return self.keypoints.get(~mask)

    def associate_points(self, indices, point_indices):
        self.point_indices[indices] = point_indices
