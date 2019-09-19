from autograd import numpy as np
from vitamine.utils import indices_other_than
from vitamine.visual_odometry.keypoint import KeypointManager
from vitamine.visual_odometry.pose import PoseManager
from vitamine.visual_odometry.timestamp import TimeStamp


class Keyframes(object):
    def __init__(self):
        self.keypoint_manager = KeypointManager()
        self.pose_manager = PoseManager()

        # leftmost is the oldest
        self.active_timestamps = []
        self.timestamp = TimeStamp()

    def add(self, keypoints, descriptors, R, t):
        self.keypoint_manager.add(keypoints, descriptors)
        self.pose_manager.add(R, t)

        timestamp = self.timestamp.get()
        self.active_timestamps.append(timestamp)
        self.timestamp.increment()
        return timestamp

    @property
    def oldest_timestamp(self):
        # Returns the oldest keyframe id in the window
        return self.active_timestamps[0]

    def id_to_index(self, timestamp):
        return timestamp - self.oldest_timestamp

    def get_keypoints(self, timestamp, indices=slice(None, None, None)):
        i = self.id_to_index(timestamp)
        return self.keypoint_manager.get(i, indices)

    def get_pose(self, timestamp):
        i = self.id_to_index(timestamp)
        return self.pose_manager.get(i)

    def get_active_poses(self):
        poses = [self.get_pose(i) for i in self.active_timestamps]
        rotations, translations = zip(*poses)
        return np.array(rotations), np.array(translations)

    def get_untriangulated(self, timestamp, triangulated_indices):
        """
        Get keypoints that have not been used for triangulation.
        These keypoints don't have corresponding 3D points.
        """
        i = self.id_to_index(timestamp)
        size = self.keypoint_manager.size(i)
        return indices_other_than(size, triangulated_indices)

    @property
    def n_active(self):
        return len(self.active_timestamps)

    def select_removed(self):
        return 0

    def remove(self):
        index = self.select_removed()
        return self.active_timestamps.pop(index)
