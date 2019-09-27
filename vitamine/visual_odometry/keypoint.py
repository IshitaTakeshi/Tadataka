from collections import namedtuple

from autograd import numpy as np


# just to enable accessing by name
# ex. localfeatures.triangulated.descriptors
KD = namedtuple("KeypointDescriptor", ["keypoints", "descriptors"])


def init_point_indices(size):
    return -np.ones(size, dtype=np.int64)


def is_triangulated(point_indices):
    return point_indices >= 0


class LocalFeatures(object):
    def __init__(self, keypoints, descriptors):
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

    def associate_points(self, indices, point_indices):
        untriangulated = np.where(~self.is_triangulated)[0]
        self.point_indices[untriangulated[indices]] = point_indices


def associate_points(lf0, lf1, matches01, point_indices):
    lf0.associate_points(matches01[:, 0], point_indices)
    lf1.associate_points(matches01[:, 1], point_indices)


def copy_point_indices(local_features_src, local_features_dst, matches01):
    point_indices = local_features_src.point_indices[matches01[:, 0]]
    local_features_dst.associate_points(matches01[:, 1], point_indices)
