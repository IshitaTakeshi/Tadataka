from collections import deque

from autograd import numpy as np

from vitamine.exceptions import (
    InvalidDepthsException, NotEnoughInliersException, print_error)
from vitamine.keypoints import extract_keypoints, Matcher
from vitamine.keypoints import KeypointDescriptor as KD
from vitamine.camera_distortion import CameraModel
from vitamine.pose import Pose
from vitamine.visual_odometry.point import Points
from vitamine.visual_odometry.pose import estimate_pose
from vitamine.visual_odometry.keypoint import init_point_indices
from vitamine.visual_odometry.triangulation import (
    triangulation, copy_triangulated, pose_point_from_keypoints)
from vitamine.visual_odometry.keyframe_index import KeyframeIndices
from vitamine.so3 import rodrigues


def get_correspondences(matcher, active_features, lf0):
    # accumulate indices of all triangulated points
    # that can be matched with lf0's descriptors
    kd0 = lf0.get()

    point_indices = []
    keypoints0_matched = []
    for lf1 in active_features:
        matches01 = matcher(kd0, lf1.triangulated())

        if len(matches01) == 0:
            continue

        p = lf1.triangulated_point_indices(matches01[:, 1])

        point_indices.append(p)
        keypoints0_matched.append(kd0.keypoints[matches01[:, 0]])

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
        omega, t = PE.solve_pnp(points_, keypoints)
    except NotEnoughInliersException:
        return None

    return Pose(omega, t)


def get_array_len_geq(min_length):
    return lambda array: len(array) >= min_length


def reprojection_error(pose, points, keypoints_true):
    from vitamine.camera import CameraParameters
    from vitamine.projection import PerspectiveProjection
    from vitamine.rigid_transform import transform
    camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])
    projection = PerspectiveProjection(camera_parameters)
    keypoints_pred = projection.compute(transform(pose.R, pose.t, points))
    return squared_norm(keypoints_true-keypoints_pred)


class VisualOdometry(object):
    def __init__(self, camera_parameters, distortion_model,
                 matcher=Matcher(enable_ransac=True),
                 min_keypoints=8, min_active_keyframes=8, min_matches=8):
        self.matcher = matcher
        self.min_active_keyframes = min_active_keyframes
        self.camera_model = CameraModel(camera_parameters, distortion_model)
        self.keypoints_condition = get_array_len_geq(min_keypoints)
        self.inlier_condition = get_array_len_geq(min_matches)
        self.active_indices = KeyframeIndices()
        self.points = Points()
        self.keypoint_descriptor_list = []
        self.point_indices_list = []
        self.poses = []

    def export_points(self):
        return self.points.get()

    def export_poses(self):
        return [[pose.omega, pose.t] for pose in self.poses]

    def add(self, image):
        return self.try_add(extract_keypoints(image))

    def init_first(self, kd, point_indices):
        self.keypoint_descriptor_list.append(kd)
        self.point_indices_list.append(point_indices)
        self.poses.append(Pose.identity())

    def try_init_second(self, kd1, point_indices1):
        kd0 = self.keypoint_descriptor_list[0]
        matches01 = self.matcher(kd0, kd1)

        if not self.inlier_condition(matches01):
            print_error("Not enough matches found")
            return False

        try:
            pose1, points, matches01 = pose_point_from_keypoints(
                kd0.keypoints, kd1.keypoints, matches01
            )
        except InvalidDepthsException as e:
            print_error(str(e))
            return False

        self.keypoint_descriptor_list.append(kd1)
        self.poses.append(pose1)
        point_indices = self.points.add(points)
        point_indices0 = self.point_indices_list[0]
        point_indices0[matches01[:, 0]] = point_indices
        point_indices1[matches01[:, 1]] = point_indices
        self.point_indices_list.append(point_indices1)
        return True

    def get_active(self, array):
        return [array[i] for i in self.active_indices]

    def try_add_more(self, kd0, point_indices0):
        active_kds = self.get_active(self.keypoint_descriptor_list)
        matches = [self.matcher(kd0, kd1) for kd1 in active_kds]
        active_point_indices = self.get_active(self.point_indices_list)
        pose0 = estimate_pose(self.points, matches,
                              active_point_indices, kd0.keypoints)
        if pose0 is None:  # pose could not be estimated
            return False

        # if not self.pose_condition(active_poses[-1], pose1):
        #     # if pose1 is too close from the latest active pose
        #     return None

        active_poses = self.get_active(self.poses)

        # copy existing point indices
        copy_triangulated(matches, active_point_indices, point_indices0)

        active_keypoints = [kd.keypoints for kd in active_kds]

        try:
            triangulation(self.points, matches,
                          active_poses, active_keypoints, active_point_indices,
                          pose0, kd0.keypoints, point_indices0)
        except InvalidDepthsException as e:
            print_error(str(e))
            return False

        self.point_indices_list.append(point_indices0)
        self.keypoint_descriptor_list.append(kd0)
        self.poses.append(pose0)
        return True

    @property
    def n_active_keyframes(self):
        return len(self.active_indices)

    def try_add(self, kd):
        keypoints, descriptors = kd

        if not self.keypoints_condition(keypoints):
            return False

        keypoints = self.camera_model.undistort(keypoints)
        return self.try_add_keyframe(KD(keypoints, descriptors))

    def try_add_keyframe(self, kd):
        point_indices = init_point_indices(len(kd.keypoints))

        if self.n_active_keyframes == 0:
            self.init_first(kd, point_indices)
            self.active_indices.add_new()
            return True

        if self.n_active_keyframes == 1:
            success = self.try_init_second(kd, point_indices)
            if not success:
                return False
            self.active_indices.add_new()
            return True

        success = self.try_add_more(kd, point_indices)
        if not success:
            return False
        self.active_indices.add_new()
        return True

    def try_remove(self):
        if self.n_active_keyframes <= self.min_active_keyframes:
            return False

        self.active_indices.remove(0)
        return True
