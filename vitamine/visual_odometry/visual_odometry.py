from collections import deque

from autograd import numpy as np

from vitamine.exceptions import (
    InvalidDepthsException, NotEnoughInliersException, print_error)
from vitamine.keypoints import extract_keypoints, match
from vitamine.camera_distortion import CameraModel
from vitamine.pose import Pose
from vitamine import pose_estimation as PE
from vitamine.visual_odometry.point import Points
from vitamine.visual_odometry.keypoint import (
    LocalFeatures, associate_points, copy_point_indices,
    init_point_indices, is_triangulated)
from vitamine.visual_odometry.triangulation import (
    triangulation, copy_triangulated, pose_point_from_keypoints)
from vitamine.visual_odometry.keyframe_index import KeyframeIndices
from vitamine.so3 import rodrigues


def find_best_match(matcher, active_descriptors, descriptors1):
    matchesx1 = [matcher(d0, descriptors1) for d0 in active_descriptors]
    argmax = np.argmax([len(m) for m in matchesx1])
    return matchesx1[argmax], argmax


def get_correspondences(matcher, active_features, lf0):
    keypoints0, descriptors0 = lf0.get()

    point_indices = []
    keypoints0_matched = []
    for lf1 in active_features:
        descriptors1 = lf1.triangulated().descriptors
        matches01 = matcher(descriptors0, descriptors1)
        if len(matches01) == 0:
            continue

        p = lf1.triangulated_point_indices(matches01[:, 1])

        point_indices.append(p)
        keypoints0_matched.append(keypoints0[matches01[:, 0]])

    if len(point_indices) == 0:
        raise NotEnoughInliersException("No matches found")

    point_indices = np.concatenate(point_indices)
    keypoints0_matched = np.vstack(keypoints0_matched)
    return point_indices, keypoints0_matched


def estimate_pose(matcher, points, active_features, lf0):
    point_indices, keypoints = get_correspondences(
        matcher, active_features, lf0
    )
    points_ = points.get(point_indices)

    try:
        omega, t = PE.solve_pnp(points_, keypoints,
                                inital_pose.omega,
                                inital_pose.translation)
    except NotEnoughInliersException:
        return None

    return Pose(omega, t)


def get_array_len_geq(min_length):
    return lambda array: len(array) >= min_length


class VisualOdometry(object):
    def __init__(self, camera_parameters, distortion_model, matcher=match,
                 min_keypoints=8, min_active_keyframes=8, min_matches=8):
        self.matcher = match
        self.min_active_keyframes = min_active_keyframes
        self.camera_model = CameraModel(camera_parameters, distortion_model)
        self.keypoints_condition = get_array_len_geq(min_keypoints)
        self.inlier_condition = get_array_len_geq(min_matches)
        self.active_indices = KeyframeIndices()
        self.points = Points()
        self.local_features = []
        self.poses = []

    def export_points(self):
        return self.points.get()

    def export_poses(self):
        return [(pose.R, pose.t) for pose in self.poses]

    def add(self, image):
        keypoints, descriptors = extract_keypoints(image)
        return self.try_add(keypoints, descriptors)

    def init_first(self, local_features):
        self.local_features.append(local_features)
        self.poses.append(Pose.identity())

    def try_init_second(self, lf1):
        lf0 = self.local_features[0]

        keypoints0, descriptors0 = lf0.get()
        keypoints1, descriptors1 = lf1.get()

        matches01 = self.matcher(descriptors0, descriptors1)
        if not self.inlier_condition(matches01):
            print_error("Not enough matches found")
            return False

        try:
            pose1, points, matches01 = pose_point_from_keypoints(
                keypoints0, keypoints1, matches01
            )
        except InvalidDepthsException as e:
            print_error(str(e))
            return False

        # if not pose_condition(pose0, pose1):
        #     return False

        self.local_features.append(lf1)
        self.poses.append(pose1)
        point_indices = self.points.add(points)
        associate_points(lf0, lf1, matches01, point_indices)
        return True

    @property
    def active_local_features(self):
        return [self.local_features[i] for i in self.active_indices]

    @property
    def active_poses(self):
        return [self.poses[i] for i in self.active_indices]

    def try_add_more(self, lf1):
        active_features = self.active_local_features

        pose1 = estimate_pose(self.matcher, self.points, active_features, lf1)
        if pose1 is None:  # pose could not be estimated
            return None

        # if not self.pose_condition(active_poses[-1], pose1):
        #     # if pose1 is too close from the latest active pose
        #     return None

        triangulation(self.matcher, self.points,
                      self.active_poses, active_features, pose1, lf1)

        # copy triangulated point indices back to lf1
        copy_triangulated(self.matcher, active_features, lf1)

        self.local_features.append(lf1)
        self.poses.append(pose1)
        return True

    @property
    def n_active_keyframes(self):
        return len(self.active_indices)

    def try_add(self, keypoints, descriptors):
        if not self.keypoints_condition(keypoints):
            return False

        keypoints = self.camera_model.undistort(keypoints)
        return self.try_add_keyframe(LocalFeatures(keypoints, descriptors))

    def try_add_keyframe(self, local_features):
        if self.n_active_keyframes == 0:
            self.init_first(local_features)
            self.active_indices.add_new()
            return True

        if self.n_active_keyframes == 1:
            success = self.try_init_second(local_features)
            if not success:
                return False
            self.active_indices.add_new()
            return True

        success = self.try_add_more(local_features)
        if not success:
            return False
        self.active_indices.add_new()
        return True

    def try_remove(self):
        if self.n_active_keyframes <= self.min_active_keyframes:
            return False

        self.active_indices.remove(0)
        return True
