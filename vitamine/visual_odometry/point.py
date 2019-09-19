from autograd import numpy as np


class PointManager(object):
    def __init__(self):
        self.visible_from = []
        self.matches = []
        self.points = []

    @property
    def n_added(self):
        # number of 'add' called so far
        return len(self.visible_from)

    def add(self, points, timestamps, matches):
        assert(len(timestamps) == 2)
        # self.points[i] is a 3D point estimated from
        # keypoints[timestamp1][matches[i, 0]] and
        # keypoints[timestamp2][matches[i, 1]]

        self.points.append(points)
        self.visible_from.append(timestamps)
        self.matches.append(matches)

    def get_points(self):
        if len(self.points) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack(self.points)

    def get(self, i):
        """
        Return 'i'-th added points, keyframe ids used for triangulation,
        and the corresponding matches
        """
        return self.points[i], self.visible_from[i], self.matches[i]

    def get_triangulated_indices(self, timestamp):
        """Get keypoint indices that already have corresponding 3D points"""
        visible_from = np.array(self.visible_from)
        frame_indices, col_indices = np.where(visible_from==timestamp)
        indices = []
        for frame_index, col_index in zip(frame_indices, col_indices):
            matches = self.matches[frame_index]
            indices.append(matches[:, col_index])
        return np.concatenate(indices)
