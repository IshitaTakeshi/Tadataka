from collections import deque

from autograd import numpy as np

from vitamine.keypoints import extract_keypoints, match
from vitamine.camera_distortion import CameraModel
from vitamine.triangulation import pose_point_from_keypoints
from vitamine.visual_odometry.pose import Pose, estimate_pose
from vitamine.visual_odometry.point import Points
from vitamine.visual_odometry.keypoint import Keypoints
from vitamine.visual_odometry.triangulation import Triangulation



def find_best_match(matcher, active_descriptors, descriptors1):
    matchesx1 = [matcher(d0, descriptors1) for d0 in active_descriptors]
    argmax = np.argmax([len(m) for m in matchesx1])
    return matchesx1[argmax], argmax


def init_points_(keypoints0, keypoints1, matches01):
    R, t, points, valid_depth_mask = pose_point_from_keypoints(
        keypoints0[matches01[:, 0]],
        keypoints1[matches01[:, 1]]
    )
    return R, t, points[valid_depth_mask], matches01[valid_depth_mask]


def init_points(matcher, keypoints0, keypoints1,
                descriptors0, descriptors1):
    matches01 = matcher(descriptors0, descriptors1)
    return init_points_(keypoints0, keypoints1, matches01)


def associate_points(keyframe0, keyframe1, matches01, point_indices):
    keyframe0.associate_points(matches01[:, 0], point_indices)
    keyframe1.associate_points(matches01[:, 1], point_indices)


def triangulation(matcher, points, descriptors_, keyframe1):
    triangulator = Triangulation(keyframe1.R, keyframe1.t,
                                 keyframe1.keypoints,
                                 keyframe1.descriptors)

    for descriptors0 in descriptors_:
        points_, matches01 = triangulator.triangulate(
            keyframe0.R, keyframe0.t,
            keypoints0, descriptors0
        )
        if len(matches01) == 0:
            continue

        point_indices = points.add(points_)
        associate_points(keyframe0, keyframe1, matches01, point_indices)


class VisualOdometry(object):
    def __init__(self, camera_parameters, distortion_model, matcher=match,
                 min_keypoints=8, min_active_keyframes=8, min_inliers=8):
        self.min_inliers = min_inliers
        self.matcher = match
        self.min_keypoints = min_keypoints
        self.min_active_keyframes = min_active_keyframes
        self.camera_model = CameraModel(camera_parameters, distortion_model)
        self.points = Points()
        self.keypoints = []
        self.poses = []

    def export_points(self):
        return self.points.get()

    def export_poses(self):
        return self.keyframes.get_poses()

    def inlier_condition(self, matches):
        return len(matches) >= self.min_inliers

    def add(self, image):
        keypoints, descriptors = extract_keypoints(image)
        return self.try_add(keypoints, descriptors)

    def try_initialize_from_two(self, keypoints1, descriptors1):
        keypoints0, descriptors0 = self.keypoints[0].get()
        R1, t1, points, matches01 = init_points(
            self.matcher,
            keypoints0, keypoints1,
            descriptors0, descriptors1
        )
        if not self.can_add_keyframe(R1, t1, points, matches01):
            return False

        kp0 = self.keypoints[0]
        kp1 = Keypoints(keypoints1, descriptors1)
        self.keypoints.append(kp1)
        point_indices = self.points.add(points)
        associate_points(kp0, kp1, matches01, point_indices)
        return True

    def try_continue(self, keypoints1, descriptors1):
        active_keyframes = self.keyframes.get_active()

        descriptors_ = [kf.triangulated()[1] for kf in active_keyframes]
        matches01, index = find_best_match(matcher, descriptors_, descriptors1)
        keyframe0 = active_keyframes[argmax]

        R1, t1 = estimate_pose(keyframe0.get_point_indices(),
                               keypoints, matches01, self.points)

        if not self.can_add_keyframe(R1, t1, points, matches01):
            return False

        self.keyframes.add(keyframe1)
        descriptors_ = [kf.untriangulated()[1] for kf in active_keyframes]
        matches01, index = find_best_match(matcher, descriptors_, descriptors1)
        triangulation(self.matcher, self.points, descriptors_, keyframe1)
        return True

    def init_first_keyframe(self, keypoints, descriptors):
        R, t = np.identity(3), np.zeros(3)
        self.keypoints.append(Keypoints(keypoints, descriptors))
        self.poses.append(Pose(R, t))

    @property
    def n_active_keyframes(self):
        return len(self.keypoints)

    def try_add(self, keypoints, descriptors):
        if len(keypoints) < self.min_keypoints:
            return False

        keypoints = self.camera_model.undistort(keypoints)

        if len(self.n_active_keyframes) == 0:
            self.init_first_keyframe(keypoints, descriptors)
            return True

        if len(self.n_active_keyframes) == 1:
            return self.try_initialize_from_two(keypoints, descriptors)
        return self.try_continue(keypoints, descriptors)

    def can_add_keyframe(self, R, t, points, matches):
        return self.inlier_condition(matches)
               # and self.pose_condition(R, t, points))

    def try_remove(self):
        if self.keyframes.active_size <= self.min_active_keyframes:
            return False

        self.keyframes.remove(self.keyframes.oldest_keyframe_id)
        return True
