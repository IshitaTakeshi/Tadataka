from autograd import numpy as np

from vitamine.keypoints import KeypointDescriptor as KD


def init_point_indices(size):
    return -np.ones(size, dtype=np.int64)


def is_triangulated(point_indices):
    return point_indices >= 0


class LocalFeatures(object):
    def __init__(self, keypoints, descriptors):
        assert(len(keypoints) == len(descriptors))
        self.keypoints = keypoints
        self.descriptors = descriptors
        # -1 for untriangulated
        self.point_indices = init_point_indices(len(keypoints))

    def get(self, indices_or_mask=None):
        if indices_or_mask is None:
            return KD(self.keypoints, self.descriptors)
        return KD(self.keypoints[indices_or_mask],
                  self.descriptors[indices_or_mask])

    @property
    def is_triangulated(self):
        return is_triangulated(self.point_indices)

    def triangulated(self):
        """
        Get keypoints that have been used for triangulation.
        """
        return self.get(self.is_triangulated)

    def untriangulated(self):
        """
        Get keypoints that have not been used for triangulation.
        These keypoints don't have corresponding 3D points.
        """
        return self.get(~self.is_triangulated)

    def triangulated_point_indices(self, indices):
        point_indices = self.point_indices[self.is_triangulated]
        return point_indices[indices]

    def associate_points(self, indices, point_indices):
        # select only untriangulated elements
        untriangulated = np.where(~self.is_triangulated)[0]
        self.point_indices[untriangulated[indices]] = point_indices
