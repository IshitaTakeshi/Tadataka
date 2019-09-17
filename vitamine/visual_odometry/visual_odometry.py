from collections import deque
from autograd import numpy as np
from vitamine.keypoints import extract_keypoints, match
from vitamine.triangulation import pose_point_from_keypoints, points_from_known_poses
from vitamine.camera_distortion import CameraModel
from vitamine.pose_estimation import solve_pnp
from vitamine.so3 import rodrigues


class Keyframes(object):
    def __init__(self):
        self.keypoint_manager = KeypointManager()
        self.pose_manager = PoseManager()

        # leftmost is the oldest
        self.active_keyframe_ids = deque()
        self.current_keyframe_id = -1

    def add(self, keypoints, descriptors, R, t):
        self.keypoint_manager.add(keypoints, descriptors)
        self.pose_manager.add(R, t)

        self.current_keyframe_id += 1
        self.active_keyframe_ids.append(self.current_keyframe_id)
        return self.current_keyframe_id

    @property
    def oldest_keyframe_id(self):
        # Returns the oldest keyframe id in the window
        return self.active_keyframe_ids[0]

    def id_to_index(self, keyframe_id):
        return keyframe_id - self.oldest_keyframe_id

    def get_keypoints(self, keyframe_id, indices=slice(None, None, None)):
        i = self.id_to_index(keyframe_id)
        return self.keypoint_manager.get(i, indices)

    def get_pose(self, keyframe_id):
        i = self.id_to_index(keyframe_id)
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
        i = self.id_to_index(keyframe_id)
        size = self.keypoint_manager.size(i)
        return indices_other_than_1d(size, triangulated_indices)

    @property
    def n_active_frames(self):
        return len(self.active_keyframe_ids)


class PoseManager(object):
    def __init__(self):
        self.rotations = []
        self.translations = []

    def add(self, R, t):
        self.rotations.append(R)
        self.translations.append(t)

    def get(self, i):
        R = self.rotations[i]
        t = self.translations[i]
        return R, t

    def get_motion_matrix(self, i):
        R, t = self.get(i)
        return motion_matrix(R, t)


class KeypointManager(object):
    def __init__(self):
        self.keypoints = []
        self.descriptors = []

    def add(self, keypoints, descriptors):
        self.keypoints.append(keypoints)
        self.descriptors.append(descriptors)

    def get(self, i, indices):
        keypoints = self.keypoints[i]
        descriptors = self.descriptors[i]
        return keypoints[indices], descriptors[indices]

    def size(self, i):
        return len(self.keypoints[i])


def indices_other_than_1d(size, indices):
    """
    size: size of the array you want to get elements from
    example:
    >>> indices_other_than_1d(8, [1, 2, 3])
    [0, 4, 5, 6, 7]
    """
    return np.setxor1d(indices, np.arange(size))


class PointManager(object):
    def __init__(self):
        self.visible_from = []
        self.matches = []
        self.points = []

    @property
    def n_frames_added(self):
        # number of 'add' called so far
        return len(self.visible_from)

    def add(self, points, keyframe_ids, matches):
        assert(len(keyframe_ids) == 2)
        # self.points[i] is a 3D point estimated from
        # keypoints[keyframe_id1][matches[i, 0]] and
        # keypoints[keyframe_id2][matches[i, 1]]

        self.matches.append(matches)
        self.points.append(points)
        self.visible_from.append(keyframe_ids)

    def get_points(self):
        return np.vstack(self.points)

    def get(self, i):
        """
        Return 'i'-th added points, keyframe ids used for triangulation,
        and the corresponding matches
        """
        return self.points[i], self.visible_from[i], self.matches[i]

    def get_triangulated_indices(self, keyframe_id):
        """Get keypoint indices that are already have corresponding 3D points"""
        visible_from = np.array(self.visible_from)
        frame_indices, col_indices = np.where(visible_from==keyframe_id)
        indices = []
        for frame_index, col_index in zip(frame_indices, col_indices):
            matches = self.matches[frame_index]
            indices.append(matches[:, col_index])
        return np.concatenate(indices)


def initialize(keypoints0, keypoints1, descriptors0, descriptors1):
    matches01 = match(descriptors0, descriptors1)

    R1, t1, points, valid_depth_mask = pose_point_from_keypoints(
        keypoints0[matches01[:, 0]],
        keypoints1[matches01[:, 1]]
    )

    return R1, t1, matches01[valid_depth_mask], points[valid_depth_mask]


class Triangulation(object):
    def __init__(self, R1, R2, t1, t2):
        self.R1, self.R2 = R1, R2
        self.t1, self.t2 = t1, t2

    def triangulate(self, keypoints1, keypoints2, descriptors1, descriptors2):
        matches12 = match(descriptors1, descriptors2)

        points, valid_depth_mask = points_from_known_poses(
            self.R1, self.R2, self.t1, self.t2,
            keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]],
        )
        return matches12[valid_depth_mask], points[valid_depth_mask]


class VisualOdometry(object):
    def __init__(self, camera_parameters, distortion_model, n_min_keypoints=8):
        self.n_min_keypoints = n_min_keypoints
        self.camera_model = CameraModel(camera_parameters, distortion_model)
        self.keyframes = Keyframes()
        self.point_manager = PointManager()

    def add(self, image):
        keypoints, descriptors = extract_keypoints(image)
        return self.try_add(keypoints, descriptors)

    def try_add(self, keypoints, descriptors):
        if len(keypoints) < self.n_min_keypoints:
            return False

        keypoints = self.camera_model.undistort(keypoints)

        if self.keyframes.n_active_frames == 0:
            self.init_keypoints(keypoints, descriptors)
            return

        keyframe_id0 = self.keyframes.oldest_keyframe_id

        if self.point_manager.n_frames_added == 0:
            success = self.try_init_points(keypoints, descriptors,
                                           keyframe_id0)
            return success

        success = self.try_add_new(keypoints, descriptors, keyframe_id0)
        return success

    def try_init_points(self, keypoints1, descriptors1, keyframe_id0):
        keypoints0, descriptors0 = self.keyframes.get_keypoints(keyframe_id0)

        R1, t1, matches01, points = initialize(keypoints0, keypoints1,
                                               descriptors0, descriptors1)

        # if not self.inlier_condition(matches):
        #     raise ValueError("No sufficient inliers found")
        # if not pose_condition(R1, t1, points):
        #     return False

        keyframe_id1 = self.keyframes.add(keypoints1, descriptors1, R1, t1)
        self.point_manager.add(points, (keyframe_id0, keyframe_id1), matches01)
        return True

    @property
    def points(self):
        # temporarl way to get points
        return self.point_manager.get_points()

    @property
    def poses(self):
        return self.keyframes.get_active_poses()

    def init_keypoints(self, keypoints, descriptors):
        R, t = np.identity(3), np.zeros(3)
        keyframe_id = self.keyframes.add(keypoints, descriptors, R, t)

    def get_untriangulated(self, keyframe_id):
        indices = self.point_manager.get_triangulated_indices(keyframe_id)
        return self.keyframes.get_untriangulated(keyframe_id, indices)

    def triangulate_new(self, keypoints1, descriptors1, R1, t1, keyframe_id0):
        # get untriangulated keypoint indices
        untriangulated_indices0 = self.get_untriangulated(keyframe_id0)

        if len(untriangulated_indices0) == 0:
            # no points to add
            return (np.empty((0, 2), dtype=np.int64),
                    np.empty((0, 3), dtype=np.float64))

        # match and triangulate with newly observed points
        keypoints0, descriptors0 = self.keyframes.get_keypoints(
            keyframe_id0, untriangulated_indices0
        )
        R0, t0 = self.keyframes.get_pose(keyframe_id0)

        triangulation = Triangulation(R0, R1, t0, t1)
        matches01, points = triangulation.triangulate(
            keypoints0, keypoints1,
            descriptors0, descriptors1
        )
        matches01[:, 0] = untriangulated_indices0[matches01[:, 0]]

        return matches01, points

    def match_existing(self, keypoints1, descriptors1, timestamp):
        """
        Match with descriptors that already have corresponding 3D points
        """

        points0, keyframe_ids, matches = self.point_manager.get(timestamp)
        # get descriptors already matched
        _, descriptorsa = self.keyframes.get_keypoints(
            keyframe_ids[0], matches[:, 0]
        )
        _, descriptorsb = self.keyframes.get_keypoints(
            keyframe_ids[1], matches[:, 1]
        )
        matches1a = match(descriptors1, descriptorsa)
        matches1b = match(descriptors1, descriptorsb)

        if len(matches1a) > len(matches1b):
            return keypoints1[matches1a[:, 0]], points0[matches1a[:, 1]]
        else:
            return keypoints1[matches1b[:, 0]], points0[matches1b[:, 1]]

    def estimate_pose(self, keypoints1, descriptors1, timestamp0):
        keypoints1_matched, points = self.match_existing(
            keypoints1, descriptors1, timestamp0,
        )
        print(keypoints1_matched.shape, keypoints1_matched.dtype)
        print(points.shape, points.dtype)
        omega1, t1 = solve_pnp(points, keypoints1_matched)
        R1 = rodrigues(omega1.reshape(1, -1))[0]
        return R1, t1

    def try_add_new(self, keypoints1, descriptors1, keyframe_id0):
        R1, t1 = self.estimate_pose(keypoints1, descriptors1, keyframe_id0)
        # if not pose_condition(R, t, points):
        #     return False
        keyframe_id1 = self.keyframes.add(keypoints1, descriptors1, R1, t1)

        matches01, points = self.triangulate_new(keypoints1, descriptors1,
                                                 R1, t1, keyframe_id0)
        if len(matches01) == 0:
            return True

        self.point_manager.add(points, (keyframe_id0, keyframe_id1), matches01)
        return True

    def remove_keyframe(self):
        self.keyframes
