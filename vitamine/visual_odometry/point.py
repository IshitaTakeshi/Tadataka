from autograd import numpy as np


def delete_multiple(array, indices):
    indices = set(indices)
    return [array[i] for i in range(len(array)) if i not in indices]


class PointManager(object):
    def __init__(self):
        self.visible_from = []
        self.matches = []
        self.points = []

    @property
    def n_added(self):
        # number of 'add' called so far
        return len(self.visible_from)

    def add(self, points, keyframe_ids, matches):
        assert(len(keyframe_ids) == 2)
        # self.points[i] is a 3D point estimated from
        # keypoints[keyframe_id1][matches[i, 0]] and
        # keypoints[keyframe_id2][matches[i, 1]]

        self.points.append(points)
        self.visible_from.append(keyframe_ids)
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

    def get_triangulated(self, keyframe_id):
        """Get keypoint indices that already have corresponding 3D points"""
        visible_from = np.array(self.visible_from)
        frame_indices, col_indices = np.where(visible_from==keyframe_id)

        if len(frame_indices) == 0:
            return np.array([], dtype=np.int64)

        indices = []
        for frame_index, col_index in zip(frame_indices, col_indices):
            matches = self.matches[frame_index]
            # 'matches[:, col_index]' have keypoints indices
            # corresponding to 'keyframe_id' that are already triangulated
            indices.append(matches[:, col_index])
        return np.concatenate(indices)

    def remove(self, keyframe_id):
        """Remove points which 'keyframe_id' is used for triangulation"""
        visible_from = np.array(self.visible_from)
        indices = np.where(visible_from==keyframe_id)[0]
        self.points = delete_multiple(self.points, indices)
        self.visible_from = delete_multiple(self.visible_from, indices)
        self.matches = delete_multiple(self.matches, indices)
