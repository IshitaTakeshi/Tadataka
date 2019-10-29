from autograd import numpy as np

from vitamine.exceptions import NotEnoughInliersException, print_error
from vitamine.keypoints import extract_keypoints, Matcher
from vitamine.keypoints import KeypointDescriptor as KD
from vitamine.camera_distortion import CameraModel
from vitamine.points import PointManager
from vitamine.pose import Pose, solve_pnp
from vitamine.keyframe_index import KeyframeIndices
from vitamine.so3 import rodrigues
from vitamine.local_ba import try_run_ba


def get_array_len_geq(min_length):
    return lambda array: len(array) >= min_length


def accumulate_correspondences(point_manager, keypoints, matches, viewpoints):
    points = []
    keypoints_ = []
    for matches01, viewpoint in zip(matches, viewpoints):
        for index0, index1 in matches01:
            try:
                point = point_manager.get(viewpoint, index0)
            except KeyError as e:
                print_error(e)
                continue
            keypoint = keypoints[index1]

            points.append(point)
            keypoints_.append(keypoint)
    return np.array(points), np.array(keypoints_)


def match(matcher, viewpoints, kds, kd1):
    matches = []
    viewpoints_ = []
    for kd0, viewpoint in zip(kds, viewpoints):
        matches01 = matcher(kd0, kd1)

        if len(matches01) == 0:
            continue

        matches.append(matches01)
        viewpoints_.append(viewpoint)
    return matches, viewpoints_


class VisualOdometry(object):
    def __init__(self, camera_parameters, distortion_model,
                 matcher=Matcher(enable_ransac=True,
                                 enable_homography_filter=True),
                 min_keypoints=8, min_active_keyframes=8, min_matches=8):
        self.matcher = matcher
        self.min_active_keyframes = min_active_keyframes
        self.camera_model = CameraModel(camera_parameters, distortion_model)
        self.keypoints_condition = get_array_len_geq(min_keypoints)
        self.inlier_condition = get_array_len_geq(min_matches)
        self.active_viewpoints = KeyframeIndices()
        self.point_manager = PointManager()
        self.kds = dict()
        self.poses = []

    def export_points(self):
        return np.copy(self.point_manager.points)

    def export_poses(self):
        return [[pose.omega, pose.t] for pose in self.poses]

    def add(self, image):
        return self.try_add(extract_keypoints(image))

    def try_init_first_two(self, kd0, kd1, viewpoint0, viewpoint1):
        matches01 = self.matcher(kd0, kd1)

        if not self.inlier_condition(matches01):
            raise NotEnoughInliersException("Not enough matches found")

        pose0, pose1 = self.point_manager.initialize(
            kd0.keypoints, kd1.keypoints, matches01,
            viewpoint0, viewpoint1
        )
        return pose0, pose1

    def try_add_more(self, kd1, viewpoint1,
                     active_kds, active_poses, active_viewpoints):
        matches, viewpoints = match(self.matcher, active_viewpoints,
                                    active_kds, kd1)
        if len(matches) == 0:
            raise NotEnoughInliersException("No matches found")

        points, keypoints = accumulate_correspondences(
            self.point_manager, kd1.keypoints,
            matches, viewpoints
        )

        try:
            pose1 = solve_pnp(points, keypoints)
        except NotEnoughInliersException as e:
            raise e

        # if not self.pose_condition(active_poses[-1], pose1):
        #     # if pose1 is too close from the latest active pose
        #     return None

        Z = zip(viewpoints, active_kds, active_poses, matches)
        for viewpoint0, kd0, pose0, matches01 in Z:
            self.point_manager.triangulate(
                pose0, pose1, kd0.keypoints, kd1.keypoints, matches01,
                viewpoint0, viewpoint1
            )

        return pose1

    @property
    def n_active_keyframes(self):
        return len(self.active_viewpoints)

    def try_add(self, kd):
        keypoints, descriptors = kd

        if not self.keypoints_condition(keypoints):
            return False

        keypoints = self.camera_model.undistort(keypoints)
        return self.try_add_keyframe(KD(keypoints, descriptors))

    def try_add_keyframe(self, new_kd):
        new_viewpoint = self.active_viewpoints.get_next()

        if self.n_active_keyframes == 0:
            self.kds[new_viewpoint] = new_kd
            self.active_viewpoints.add_new(new_viewpoint)
            return True

        if self.n_active_keyframes == 1:
            viewpoint0 = self.active_viewpoints[-1]
            try:
                pose0, pose1 = self.try_init_first_two(
                    kd0=self.kds[0], kd1=new_kd,
                    viewpoint0=viewpoint0, viewpoint1=new_viewpoint
                )
            except NotEnoughInliersException:
                return False

            self.poses = [pose0, pose1]
            self.kds[new_viewpoint] = new_kd
            self.active_viewpoints.add_new(new_viewpoint)
            return True

        active_kds = [self.kds[v] for v in self.active_viewpoints]
        active_poses = [self.poses[v] for v in self.active_viewpoints]
        try:
            pose1 = self.try_add_more(new_kd, new_viewpoint,
                                      active_kds, active_poses,
                                      self.active_viewpoints)
        except NotEnoughInliersException:
            return False

        self.poses.append(pose1)
        self.kds[new_viewpoint] = new_kd
        self.active_viewpoints.add_new(new_viewpoint)

        active_poses = [self.poses[v] for v in self.active_viewpoints]
        active_kds = [self.kds[v] for v in self.active_viewpoints]
        try:
            poses, points, point_indices = try_run_ba(
                self.point_manager.index_map,
                self.point_manager.points,
                active_poses,
                [kd.keypoints for kd in active_kds],
                self.active_viewpoints
            )

            for j, pose in zip(self.active_viewpoints, poses):
                self.poses[j] = pose

            assert(len(point_indices) == len(points))
            for i, point in zip(point_indices, points):
                self.point_manager.overwrite(i, point)

        except ValueError as e:
            print_error(e)
        return True

    def try_remove(self):
        if self.n_active_keyframes <= self.min_active_keyframes:
            return False

        self.active_viewpoints.remove(0)
        return True
