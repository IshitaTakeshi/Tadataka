import warnings
import numpy as np

from skimage.color import rgb2gray
from tadataka.exceptions import NotEnoughInliersException, print_error
from tadataka.features import extract_features, Matcher
from tadataka.features import Features as KD
from tadataka.camera import CameraModel
from tadataka.point_keypoint_map import (
    associate_triangulated,
    get_indices, get_point_hashes, init_correspondence, is_triangulated,
    merge_correspondences, subscribe
)
from tadataka.utils import merge_dicts, value_list
from tadataka.pose import Pose, solve_pnp, estimate_pose_change
from tadataka.triangulation import triangulate
from tadataka.keyframe_index import KeyframeIndices
from tadataka.so3 import rodrigues
from tadataka.local_ba import try_run_ba
from tadataka.visual_odometry.base import BaseVO


def get_new_viewpoint(viewpoints):
    if len(viewpoints) == 0:
        return 0
    return viewpoints[-1] + 1


def extract_colors(correspondence, point_dict, keypoints, image):
    point_colors = dict()
    keypoints = keypoints.astype(np.int64)
    for point_hash in point_dict.keys():
        keypoint_index = correspondence[point_hash]
        x, y = keypoints[keypoint_index]
        point_colors[point_hash] = image[y, x]
    return point_colors


def unique_point_hashes(correspondences):
    point_hashes = set()
    for correspondence in correspondences:
        point_hashes |= set(correspondence.keys())
    return list(point_hashes)


def get_ba_indices(correspondences, kds, point_hashes):
    assert(len(kds) == len(correspondences))

    viewpoint_indices = []
    point_indices = []
    keypoints = []
    for j, (kd, map_) in enumerate(zip(kds, correspondences)):
        for i, point_hash in enumerate(point_hashes):
            try:
                keypoint_index = map_[point_hash]
            except KeyError:
                # continue if corresponding keypoint does not exist
                continue
            viewpoint_indices.append(j)
            point_indices.append(i)
            keypoints.append(kd.keypoints[keypoint_index])

    return (np.array(viewpoint_indices), np.array(point_indices),
            np.array(keypoints))


def filter_matches(matches, viewpoints, min_matches):
    assert(len(viewpoints) == len(matches))

    def f(args):
        matches01, viewpoint = args
        return len(matches01) >= min_matches

    Z = filter(f, zip(matches, viewpoints))
    return zip(*Z)


class FeatureBasedVO(BaseVO):
    def __init__(self, camera_parameters, distortion_model,
                 matcher=Matcher(enable_ransac=True,
                                 enable_homography_filter=True),
                 window_size=8, min_matches=60):

        super().__init__(camera_parameters, distortion_model)
        self.__window_size = window_size

        self.matcher = matcher
        self.min_matches = min_matches

        self.active_viewpoints = np.empty((0, 0), np.int64)
        # manages point -> keypoint correspondences
        self.correspondences = dict()

        self.point_colors = dict()
        self.point_dict = dict()
        self.kds = dict()
        self.poses = dict()
        self.images = dict()
        self.kds_ = dict()

    def export_points(self):
        assert(len(self.point_dict) == len(self.point_colors))
        point_hashes = self.point_dict.keys()
        point_array = np.array(value_list(self.point_dict, point_hashes))
        point_colors = np.array(value_list(self.point_colors, point_hashes))
        point_colors = point_colors.astype(np.float64) / 255.
        return point_array, point_colors

    def export_poses(self):
        return [self.poses[v] for v in sorted(self.poses.keys())]

    @property
    def n_active_keyframes(self):
        return len(self.active_viewpoints)

    def init_first_two(self, kd1, viewpoint0):
        pose0 = self.poses[viewpoint0]
        kd0 = self.kds[viewpoint0]

        matches, viewpoints = self.match(kd1, viewpoints=[viewpoint0])
        if len(matches) == 0:
            raise ValueError("Not enough matches found")

        matches01, viewpoint0 = matches[0], viewpoints[0]

        pose1 = estimate_pose_change(kd0.keypoints, kd1.keypoints, matches01)
        point_array, matches01 = triangulate(pose0, pose1,
                                             kd0.keypoints, kd1.keypoints, matches01)
        point_dict, map0, map1 = subscribe(point_array, matches01)
        return pose1, point_dict, map0, map1

    def estimate_pose_points(self, kd1):
        # TODO swap if condition. this should be
        # if len(self.active_viewpoints) > 1
        #   return self.estimate_pose_points_(kd1, self.active_viewpoints)
        # ...
        if len(self.active_viewpoints) == 1:
            viewpoint0 = self.active_viewpoints[0]
            pose1, point_dict, map0, map1 = self.init_first_two(
                kd1, viewpoint0
            )
            map0s = {viewpoint0: map0}
            return pose1, point_dict, map0s, map1
        return self.estimate_pose_points_(kd1, self.active_viewpoints)

    def estimate_pose_points_(self, kd1, viewpoints):
        matches, viewpoints = self.match(kd1, viewpoints)
        pose1 = self.estime_pose(kd1, viewpoints, matches)
        point_dict, map0s, map1 = self.triangulate(
            viewpoints, matches, pose1, kd1
        )
        return pose1, point_dict, map0s, map1

    def add(self, image, min_keypoints=8):
        keypoints, descriptors = extract_features(image)

        if len(keypoints) <= min_keypoints:
            print_error("Keypoints not sufficient")
            return -1

        viewpoint1 = get_new_viewpoint(self.active_viewpoints)
        kd1 = KD(self.camera_model.undistort(keypoints), descriptors)

        if len(self.active_viewpoints) == 0:
            map1 = init_correspondence()
            pose1 = Pose.identity()
            point_dict = dict()
        else:
            try:
                pose1, point_dict, map0s, map1 = self.estimate_pose_points(kd1)
            except NotEnoughInliersException as e:
                print_error(e)
                return -1

            for viewpoint0, m0 in map0s.items():
                self.correspondences[viewpoint0] = merge_correspondences(
                    self.correspondences[viewpoint0], m0
                )

        self.poses[viewpoint1] = pose1
        self.correspondences[viewpoint1] = map1

        # use distorted (not normalized) keypoints
        point_colors = extract_colors(map1, point_dict, keypoints, image)
        self.point_colors.update(point_colors)
        self.point_dict.update(point_dict)

        self.kds[viewpoint1] = kd1
        self.images[viewpoint1] = image
        self.active_viewpoints = np.append(self.active_viewpoints, viewpoint1)

        if len(self.active_viewpoints) >= 3:
            self.run_ba(self.active_viewpoints)
        return viewpoint1

    def run_ba(self, viewpoints):
        correspondences = value_list(self.correspondences, viewpoints)
        poses = value_list(self.poses, viewpoints)
        kds = value_list(self.kds, viewpoints)

        point_hashes = unique_point_hashes(correspondences)

        point_array = np.array(value_list(self.point_dict, point_hashes))

        viewpoint_indices, point_indices, keypoints = get_ba_indices(
            correspondences, kds, point_hashes
        )

        poses, point_array = try_run_ba(viewpoint_indices, point_indices,
                                        poses, point_array, keypoints)

        for point_hash, point in zip(point_hashes, point_array):
            self.point_dict[point_hash] = point

        for viewpoint, pose in zip(viewpoints, poses):
            self.poses[viewpoint] = pose

    def estime_pose(self, kd1, viewpoints, matches):
        assert(len(viewpoints) == len(matches))
        correspondences = value_list(self.correspondences, viewpoints)

        point_hashes = []
        keypoint_indices = []
        for viewpoint, matches01 in zip(viewpoints, matches):
            correspondences = self.correspondences[viewpoint]
            hashes_, indices_ = get_indices(correspondences, matches01)
            point_hashes += hashes_
            keypoint_indices += indices_
        assert(len(point_hashes) == len(keypoint_indices))
        point_array = np.array(value_list(self.point_dict, point_hashes))
        return solve_pnp(point_array, kd1.keypoints[keypoint_indices])

    def match_(self, kd1, viewpoints):
        kds = value_list(self.kds, viewpoints)
        return [self.matcher(kd0, kd1) for kd0 in kds]

    def match(self, kd1, viewpoints):
        matches = self.match_(kd1, viewpoints)
        # select matches that have enough inliers
        return filter_matches(matches, viewpoints, self.min_matches)

    def triangulate_(self, matches01, viewpoint0, pose1, kd1):
        pose0 = self.poses[viewpoint0]
        kd0 = self.kds[viewpoint0]
        map0 = self.correspondences[viewpoint0]

        mask = is_triangulated(map0, matches01[:, 0])
        triangulated, untriangulated = matches01[mask], matches01[~mask]

        map1_copied = associate_triangulated(map0, triangulated)

        if len(untriangulated) == 0:
            return dict(), init_correspondence(), init_correspondence()

        # if point doesn't exist, create it by triangulation
        point_array, triangulated_ = triangulate(
            pose0, pose1, kd0.keypoints, kd1.keypoints, untriangulated
        )
        point_dict, map0_created, map1_created = subscribe(
            point_array, triangulated_
        )
        map1 = merge_correspondences(map1_copied, map1_created)
        return point_dict, map0_created, map1

    def triangulate(self, viewpoints, matches, pose1, kd1):
        # filter keypoints so that one keypoint has only one corresponding
        # 3D point
        used_indices1 = set()
        def filter_unused(matches01):
            matches01_ = []
            for index0, index1 in matches01:
                if index1 not in used_indices1:
                    matches01_.append([index0, index1])
                    used_indices1.add(index1)
            return np.array(matches01_)

        point_dict = dict()
        map0s = dict()
        map1 = init_correspondence()
        for viewpoint0, matches01 in zip(viewpoints, matches):
            matches01 = filter_unused(matches01)

            if len(matches01) == 0:
                continue

            point_dict_, map0_, map1_ = self.triangulate_(
                matches01, viewpoint0, pose1, kd1
            )
            map0s[viewpoint0] = map0_
            map1 = merge_correspondences(map1, map1_)
            point_dict.update(point_dict_)

        return point_dict, map0s, map1

    def try_remove(self):
        if self.n_active_keyframes <= self.__window_size:
            return False

        self.active_viewpoints = np.delete(self.active_viewpoints, 0)
        return True
