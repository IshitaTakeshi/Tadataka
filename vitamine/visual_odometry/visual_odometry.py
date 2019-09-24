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


def init_points(local_features0, local_features1, matcher, inlier_condition,
                valid_depth_ratio=0.8):

    keypoints0, descriptors0 = local_features0.get()
    keypoints1, descriptors1 = local_features1.get()

    matches01 = matcher(descriptors0, descriptors1)
    print("len(matches01)", len(matches01))
    if not inlier_condition(matches01):
        raise NotEnoughInliersException("Not enough matches found")

    R, t, points, valid_depth_mask = pose_point_from_keypoints(
        keypoints0[matches01[:, 0]],
        keypoints1[matches01[:, 1]]
    )

    if np.sum(valid_depth_mask) / len(valid_depth_mask) < valid_depth_ratio:
        raise InvalidDepthsException(
            "Most of points are behind cameras. Maybe wrong matches?"
        )

    return Pose(R, t), points[valid_depth_mask], matches01[valid_depth_mask]


def associate_points(local_features0, local_features1,
                     matches01, point_indices):
    local_features0.associate_points(matches01[:, 0], point_indices)
    local_features1.associate_points(matches01[:, 1], point_indices)


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


def get_array_len_geq(min_length):
    return lambda array: len(array) >= min_length


class VisualOdometry(object):
    def __init__(self, camera_parameters, distortion_model, matcher=match,
                 min_keypoints=8, min_active_keyframes=8, min_matches=8):
        self.matcher = match
        self.min_active_keyframes = min_active_keyframes
        self.camera_model = CameraModel(camera_parameters, distortion_model)
        self.points = Points()
        self.keypoints = []
        self.poses = []
        self.keypoints_condition = get_array_len_geq(min_keypoints)
        self.inlier_condition = get_array_len_geq(min_matches)

    def export_points(self):
        return self.points.get()

    def export_poses(self):
        return [(pose.R, pose.t) for pose in self.poses]

    def add(self, image):
        keypoints, descriptors = extract_keypoints(image)
        return self.try_add(keypoints, descriptors)

    def init_first(self, local_features):
        self.keypoints.append(local_features)
        self.poses.append(Pose.identity())

    def try_init_second(self, lf1):
        lf0 = self.keypoints[0]

        try:
            pose1, points, matches01 = init_points(lf0, lf1, self.matcher,
                                                   self.inlier_condition)
        except InvalidDepthsException as e:
            print_error(str(e))
            return False
        except NotEnoughInliersException as e:
            print_error(str(e))
            return False

        self.keypoints.append(lf1)
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

        lf = LocalFeatures(keypoints, descriptors)

        if len(self.n_active_keyframes) == 0:
            self.init_first(lf)
            return True

        if len(self.n_active_keyframes) == 1:
            return self.try_init_second(lf)
        return self.try_continue(lf)

    def try_remove(self):
        if self.keyframes.active_size <= self.min_active_keyframes:
            return False

        self.keyframes.remove(self.keyframes.oldest_keyframe_id)
        return True
