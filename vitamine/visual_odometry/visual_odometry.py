from collections import deque
from autograd import numpy as np
from vitamine.keypoints import extract_keypoints, match
from vitamine.triangulation import pose_point_from_keypoints, points_from_known_poses
from vitamine.camera_distortion import CameraModel

from vitamine.visual_odometry.pose import PoseManager, PoseEstimator
from vitamine.visual_odometry.point import PointManager
from vitamine.visual_odometry.keyframe import Keyframes


class Triangulation(object):
    def __init__(self, matcher, R1, R2, t1, t2):
        self.matcher = matcher
        self.R1, self.R2 = R1, R2
        self.t1, self.t2 = t1, t2

    def triangulate(self, keypoints1, keypoints2, descriptors1, descriptors2):
        matches12 = self.matcher(descriptors1, descriptors2)

        points, valid_depth_mask = points_from_known_poses(
            self.R1, self.R2, self.t1, self.t2,
            keypoints1[matches12[:, 0]], keypoints2[matches12[:, 1]],
        )
        return matches12[valid_depth_mask], points[valid_depth_mask]


def initialize(keypoints0, keypoints1, descriptors0, descriptors1):
    matches01 = match(descriptors0, descriptors1)

    R1, t1, points, valid_depth_mask = pose_point_from_keypoints(
        keypoints0[matches01[:, 0]],
        keypoints1[matches01[:, 1]]
    )

    return R1, t1, matches01[valid_depth_mask], points[valid_depth_mask]


class VisualOdometry(object):
    def __init__(self, camera_parameters, distortion_model, matcher=match,
                 min_keypoints=8, min_active_keyframes=8):
        self.matcher = match
        self.min_keypoints = min_keypoints
        self.min_active_keyframes = min_active_keyframes
        self.camera_model = CameraModel(camera_parameters, distortion_model)
        self.keyframes = Keyframes()
        self.point_manager = PointManager()

    def add(self, image):
        keypoints, descriptors = extract_keypoints(image)
        return self.try_add(keypoints, descriptors)

    @property
    def reference_timestamp(self):
        return self.keyframes.oldest_timestamp

    def try_add(self, keypoints, descriptors):
        if len(keypoints) < self.min_keypoints:
            return False

        keypoints = self.camera_model.undistort(keypoints)

        if self.keyframes.n_active == 0:
            self.init_keypoints(keypoints, descriptors)
            return True

        timestamp0 = self.reference_timestamp

        if self.point_manager.n_added == 0:
            success = self.try_init_points(keypoints, descriptors,
                                           timestamp0)
            return success

        success = self.try_add_new(keypoints, descriptors, timestamp0)
        return success

    def try_init_points(self, keypoints1, descriptors1, timestamp0):
        keypoints0, descriptors0 = self.keyframes.get_keypoints(timestamp0)

        initializer = Initializer(self.matcher, keypoints0, descriptors0)
        R1, t1, matches01, points = initializer.initialize(keypoints1, descriptors1)

        # TODO
        # if not self.inlier_condition(matches):
        #     return False
        # if not pose_condition(R1, t1, points):
        #     return False
        timestamp1 = self.keyframes.add(keypoints1, descriptors1, R1, t1)
        self.point_manager.add(points, (timestamp0, timestamp1), matches01)
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
        timestamp = self.keyframes.add(keypoints, descriptors, R, t)

    def get_untriangulated(self, timestamp):
        indices = self.point_manager.get_triangulated_indices(timestamp)
        return self.keyframes.get_untriangulated(timestamp, indices)

    def triangulate_new(self, keypoints1, descriptors1, R1, t1, timestamp0):
        # get untriangulated keypoint indices
        untriangulated_indices0 = self.get_untriangulated(timestamp0)

        if len(untriangulated_indices0) == 0:
            # no points to add
            return (np.empty((0, 2), dtype=np.int64),
                    np.empty((0, 3), dtype=np.float64))

        # match and triangulate with newly observed points
        keypoints0, descriptors0 = self.keyframes.get_keypoints(
            timestamp0, untriangulated_indices0
        )
        R0, t0 = self.keyframes.get_pose(timestamp0)

        triangulation = Triangulation(self.matcher, R0, R1, t0, t1)
        matches01, points = triangulation.triangulate(
            keypoints0, keypoints1,
            descriptors0, descriptors1
        )
        matches01[:, 0] = untriangulated_indices0[matches01[:, 0]]

        return matches01, points

    def get_descriptors(self, timestamp0):
        # 3D points have corresponding two viewpoits used for triangulation
        # To estimate the pose of the new frame, match keypoints in the new
        # frame to keypoints in the two viewpoints
        # Matched keypoints have corresponding 3D points.
        # Therefore we can estimate the pose of the new frame using the matched keypoints
        # and corresponding 3D points.
        points0, timestamps, matches = self.point_manager.get(timestamp0)
        # get descriptors already matched
        _, descriptors0a = self.keyframes.get_keypoints(
            timestamps[0], matches[:, 0]
        )
        _, descriptors0b = self.keyframes.get_keypoints(
            timestamps[1], matches[:, 1]
        )
        return points0, descriptors0a, descriptors0b

    def try_add_new(self, keypoints1, descriptors1, timestamp0):
        estimator = PoseEstimator(self.matcher,
                                  *self.get_descriptors(timestamp0))
        R1, t1 = estimator.estimate(keypoints1, descriptors1)
        # if not pose_condition(R, t, points):
        #     return False
        timestamp1 = self.keyframes.add(keypoints1, descriptors1, R1, t1)

        matches01, points = self.triangulate_new(keypoints1, descriptors1,
                                                 R1, t1, timestamp0)
        if len(matches01) == 0:
            return True

        self.point_manager.add(points, (timestamp0, timestamp1), matches01)
        return True

    def try_remove(self):
        if self.keyframes.n_active <= self.min_active_keyframes:
            return False

        timestamp = self.keyframes.remove()
        return True
