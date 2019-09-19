from autograd import numpy as np
from vitamine.utils import indices_other_than
from vitamine.visual_odometry.keypoint import KeypointManager
from vitamine.visual_odometry.pose import PoseManager
from vitamine.visual_odometry.timestamp import TimeStamp


class Keyframes(object):
    def __init__(self):
        self.keypoint_manager = KeypointManager()
        self.pose_manager = PoseManager()

        self.active_keyframe_ids = []  # leftmost is the oldest
        self.timestamp = TimeStamp()

    def add(self, keypoints, descriptors, R, t):
        self.keypoint_manager.add(keypoints, descriptors)
        self.pose_manager.add(R, t)

        keyframe_id = self.timestamp.get()
        self.active_keyframe_ids.append(keyframe_id)
        self.timestamp.increment()
        return keyframe_id

    @property
    def oldest_keyframe_id(self):
        # Returns the oldest keyframe id in the window
        return self.active_keyframe_ids[0]

    def keyframe_id_to_index(self, keyframe_id):
        return keyframe_id - self.oldest_keyframe_id

    def get_keypoints(self, keyframe_id, indices=slice(None, None, None)):
        i = self.keyframe_id_to_index(keyframe_id)
        return self.keypoint_manager.get(i, indices)

    def get_pose(self, keyframe_id):
        i = self.keyframe_id_to_index(keyframe_id)
        return self.pose_manager.get(i)

    def get_active_poses(self):
        poses = [self.get_pose(i) for i in self.active_keyframe_ids]
        rotations, translations = zip(*poses)
        return np.array(rotations), np.array(translations)

    def get_untriangulated(self, keyframe_id, triangulated_indices):
        """
        Get keypoints that have not been used for triangulation.
        These keypoints don't have corresponding 3D points.
        """
        i = self.keyframe_id_to_index(keyframe_id)
        size = self.keypoint_manager.size(i)
        return indices_other_than(size, triangulated_indices)

    @property
    def n_active(self):
        return len(self.active_keyframe_ids)

    def remove(self, keyframe_id):
        return self.active_keyframe_ids.remove(keyframe_id)
