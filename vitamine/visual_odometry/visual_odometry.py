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


def match_triangulate(matcher, points, lf0, lf1, pose0, pose1):
    keypoints0, descriptors0 = lf0.untriangulated()
    keypoints1, descriptors1 = lf1.untriangulated()
    matches01 = matcher(descriptors0, descriptors1)

    if len(matches01) == 0:
        return

    try:
        points_, matches01 = points_from_known_poses(
            keypoints0, keypoints1, pose0, pose1, matches01)
    except InvalidDepthsException as e:
        print_error(str(e))
        return

    # subscribe points and associate it with keypoints
    point_indices = points.add(points_)
    associate_points(lf0, lf1, matches01, point_indices)


def triangulation(matcher, points,
                  local_features_list, pose_list, lf0, pose0):
    descriptors0 = lf0.untriangulated().descriptors

    for i, (lf1, pose1) in enumerate(zip(local_features_list, pose_list)):
        match_triangulate(matcher, points, lf0, lf1, pose0, pose1)

    for lf1 in local_features_list:
        descriptors1 = lf1.untriangulated().descriptors
        matches01 = matcher(descriptors0, descriptors1)
        copy_point_indices(lf0, lf1, matches01)


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

    @property
    def n_active_keyframes(self):
        return len(self.keypoints)

    def try_add(self, keypoints, descriptors):
        if len(keypoints) < self.min_keypoints:
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
        if self.keyframes.active_size <= self.min_active_keyframes:
            return False

        self.keyframes.remove(self.keyframes.oldest_keyframe_id)
        return True
