import warnings
from autograd import numpy as np

from vitamine.exceptions import NotEnoughInliersException, print_error
from vitamine.keypoints import extract_keypoints, Matcher
from vitamine.keypoints import KeypointDescriptor as KD
from vitamine.camera_distortion import CameraModel
from vitamine.point_keypoint_map import (
    get_correspondences, get_point_hashes, init_point_keypoint_map,
    merge_point_keypoint_maps, point_exists
)
from vitamine.utils import merge_dicts
from vitamine.pose import Pose, solve_pnp, estimate_pose_change
from vitamine.triangulation import Triangulation
from vitamine.keyframe_index import KeyframeIndices
from vitamine.random import random_bytes
from vitamine.so3 import rodrigues
from skimage.color import rgb2gray
from vitamine.local_ba import try_run_ba


def get_new_viewpoint(viewpoints):
    if len(viewpoints) == 0:
        return 0
    return viewpoints[-1] + 1


def generate_hashes(n_hashes, n_bytes=18):
    return [random_bytes(n_bytes) for i in range(n_hashes)]


def point_array(point_dict, point_keys):
    return np.array([point_dict[k] for k in point_keys])


def associate_points_keypoints(point_array, matches01):
    point_hashes = generate_hashes(len(point_array))

    assert(len(point_hashes) == len(matches01))
    map0 = init_point_keypoint_map(zip(point_hashes, matches01[:, 0]))
    map1 = init_point_keypoint_map(zip(point_hashes, matches01[:, 1]))
    point_dict = dict(zip(point_hashes, point_array))
    return map0, map1, point_dict


def associate_triangulated(map0, matches01):
    point_hashes0 = get_point_hashes(map0, matches01[:, 0])
    return init_point_keypoint_map(zip(point_hashes0, matches01[:, 1]))


def triangulate(pose0, pose1, keypoints0, keypoints1, matches01):
    t = Triangulation(pose0, pose1, keypoints0, keypoints1)
    point_array, depth_mask = t.triangulate(matches01)
    # preserve points that have positive depths
    return point_array, matches01[depth_mask]


def value_list(dict_, keys):
    return [dict_[k] for k in keys]


def separate(point_keypoint_map0, matches01):
    indices0, map0 = matches01[:, 0], point_keypoint_map0
    is_triangulated = np.array([point_exists(map0, i) for i in indices0])
    return matches01[is_triangulated], matches01[~is_triangulated]


def unique_point_hashes(point_keypoint_maps):
    point_hashes = set()
    for point_keypoint_map in point_keypoint_maps:
        point_hashes |= set(point_keypoint_map.keys())
    return list(point_hashes)


def get_ba_indices(point_keypoint_maps, kds, point_hashes):
    assert(len(kds) == len(point_keypoint_maps))

    viewpoint_indices = []
    point_indices = []
    keypoints = []
    for j, (kd, map_) in enumerate(zip(kds, point_keypoint_maps)):
        for i, point_hash in enumerate(point_hashes):
            try:
                keypoint_index = map_[point_hash]
            except KeyError:
                continue
            viewpoint_indices.append(j)
            point_indices.append(i)
            keypoints.append(kd.keypoints[keypoint_index])

    return (np.array(viewpoint_indices), np.array(point_indices),
            np.array(keypoints))


def filter_matches(matches, viewpoints, min_matches):
    assert(len(viewpoints) == len(matches))
    matches_ = []
    viewpoints_ = []
    for viewpoint, matches01 in zip(viewpoints, matches):
        if len(matches01) >= min_matches:
            matches_.append(matches01)
            viewpoints_.append(viewpoint)
    return matches_, viewpoints_


class VisualOdometry(object):
    def __init__(self, camera_parameters, distortion_model,
                 matcher=Matcher(enable_ransac=True,
                                 enable_homography_filter=True),
                 max_active_keyframes=8, min_matches=60):

        self.matcher = matcher
        self.min_matches = min_matches
        self.max_active_keyframes = max_active_keyframes
        self.camera_model = CameraModel(camera_parameters, distortion_model)

        self.active_viewpoints = np.empty((0, 0), np.int64)
        self.point_keypoint_maps = dict()

        self.point_colors = dict()
        self.point_dict = dict()
        self.kds = dict()
        self.poses = dict()
        self.images = dict()
        self.kds_ = dict()

    def export_points(self):
        return np.array(list(self.point_dict.values()))

    def export_poses(self):
        poses = [self.poses[v] for v in sorted(self.poses.keys())]
        return [[pose.omega, pose.t] for pose in poses]

    @property
    def n_active_keyframes(self):
        return len(self.active_viewpoints)

    def add(self, image, min_keypoints=8):
        keypoints, descriptors = extract_keypoints(rgb2gray(image))

        if len(keypoints) <= min_keypoints:
            print_error("Keypoints not sufficient")
            return -1

        viewpoint = get_new_viewpoint(self.active_viewpoints)
        kd = KD(self.camera_model.undistort(keypoints), descriptors)

        if len(self.active_viewpoints) == 0:
            retval = self.init_first(viewpoint)
        elif len(self.active_viewpoints) == 1:
            retval = self.try_init_first_two(viewpoint, kd)
        else:
            retval = self.try_add_keyframe(viewpoint, kd)

        if retval < 0:
            return -1

        # map_ = self.point_keypoint_maps[viewpoint]
        # for point_hash in point_dict.keys():
        #     keypoint_index = map_[point_hash]
        #     y, x = keypoints[keypoint_index]
        #     self.point_colors[point_hash] = image[y, x]

        self.kds[viewpoint] = kd
        self.images[viewpoint] = image
        self.active_viewpoints = np.append(self.active_viewpoints, viewpoint)

        if len(self.active_viewpoints) >= 3:
            self.run_ba(self.active_viewpoints)
        return viewpoint

    def run_ba(self, viewpoints):
        point_keypoint_maps = value_list(self.point_keypoint_maps, viewpoints)
        poses = value_list(self.poses, viewpoints)
        kds = value_list(self.kds, viewpoints)

        point_hashes = unique_point_hashes(point_keypoint_maps)

        point_array = np.array(value_list(self.point_dict, point_hashes))

        viewpoint_indices, point_indices, keypoints = get_ba_indices(
            point_keypoint_maps, kds, point_hashes
        )

        poses, point_array = try_run_ba(viewpoint_indices, point_indices,
                                        poses, point_array, keypoints)

        for point_hash, point in zip(point_hashes, point_array):
            self.point_dict[point_hash] = point

        for viewpoint, pose in zip(viewpoints, poses):
            self.poses[viewpoint] = pose

    def init_first(self, viewpoint1):
        self.poses[viewpoint1] = Pose.identity()
        return 0

    def try_init_first_two(self, viewpoint1, kd1):
        viewpoint0 = self.active_viewpoints[0]
        kd0 = self.kds[viewpoint0]

        matches01 = self.matcher(kd0, kd1)
        if len(matches01) < self.min_matches:
            return -1

        keypoints0, keypoints1 = kd0.keypoints, kd1.keypoints

        pose0 = Pose.identity()
        pose1 = estimate_pose_change(kd0.keypoints, kd1.keypoints, matches01)

        point_array, matches01 = triangulate(
            pose0, pose1, kd0.keypoints, kd1.keypoints, matches01
        )
        map0, map1, point_dict = associate_points_keypoints(
            point_array, matches01
        )
        self.point_keypoint_maps[viewpoint0] = map0
        self.point_keypoint_maps[viewpoint1] = map1
        self.point_dict = point_dict
        self.poses[viewpoint0] = pose0
        self.poses[viewpoint1] = pose1
        return 0

    def estime_pose(self, kd1, point_keypoint_maps, matches):
        assert(len(point_keypoint_maps) == len(matches))
        point_hashes, keypoint_indices = get_correspondences(
            point_keypoint_maps, matches
        )
        point_array = np.array(value_list(self.point_dict, point_hashes))
        return solve_pnp(point_array, kd1.keypoints[keypoint_indices])

    def match_(self, kd1, viewpoints):
        kds = value_list(self.kds, viewpoints)
        return [self.matcher(kd0, kd1) for kd0 in kds]

    def match(self, kd1):
        return self.match_(kd1, self.active_viewpoints)

    def try_add_keyframe(self, viewpoint1, kd1):
        matches = self.match_(kd1, viewpoints=self.active_viewpoints)

        # select matches that have enough inliers
        matches, viewpoints = filter_matches(matches, self.active_viewpoints,
                                             self.min_matches)

        maps = value_list(self.point_keypoint_maps, viewpoints)
        pose1 = self.estime_pose(kd1, maps, matches)
        poses = value_list(self.poses, viewpoints)
        kds = value_list(self.kds, viewpoints)

        map1s = []
        for viewpoint0, map0, matches01 in zip(viewpoints, maps, matches):
            pose0 = self.poses[viewpoint0]
            kd0 = self.kds[viewpoint0]
            triangulated, untriangulated = separate(map0, matches01)
            # Find keypoints that already have corresponding 3D points
            # If keypoint in one frame has corresponding 3D point,
            # associate it to the matched keypoint in the other frame
            map1a = associate_triangulated(
                self.point_keypoint_maps[viewpoint0],
                triangulated
            )
            map1s.append(map1a)

            # if point doesn't exist, create it by triangulation
            point_array, triangulated_ = triangulate(
                pose0, pose1, kd0.keypoints, kd1.keypoints, untriangulated
            )

            map0, map1b, point_dict = associate_points_keypoints(
                point_array, triangulated_
            )
            map1s.append(map1b)
            self.point_dict.update(point_dict)

            self.point_keypoint_maps[viewpoint0].update(map0)

        map1 = merge_point_keypoint_maps(*map1s)

        self.point_keypoint_maps[viewpoint1] = map1
        self.poses[viewpoint1] = pose1
        # self.try_run_ba(self.active_viewpoints)
        return 0

    def try_remove(self):
        if self.n_active_keyframes <= self.max_active_keyframes:
            return False

        viewpoint = self.active_viewpoints.remove(0)
        print("viewpoint {} has been removed".format(viewpoint))
        return True
