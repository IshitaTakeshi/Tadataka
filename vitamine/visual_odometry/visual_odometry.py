import warnings
from autograd import numpy as np

from vitamine.exceptions import NotEnoughInliersException, print_error
from vitamine.keypoints import extract_keypoints, Matcher
from vitamine.keypoints import KeypointDescriptor as KD
from vitamine.camera_distortion import CameraModel
from vitamine.point_keypoint_map import (
    init_point_keypoint_map, get_correspondences,
    triangulation_required, copy_required, get_point_hashes,
    merge_point_keypoint_maps
)
from vitamine.utils import merge_dicts
from vitamine.pose import Pose, solve_pnp, estimate_pose_change
from vitamine.triangulation import Triangulation
from vitamine.keyframe_index import KeyframeIndices
from vitamine.random import random_bytes
from vitamine.so3 import rodrigues
from skimage.color import rgb2gray
from vitamine.local_ba import try_run_ba


def generate_hashes(n_hashes, n_bytes=18):
    return [random_bytes(n_bytes) for i in range(n_hashes)]


def point_array(point_dict, point_keys):
    return np.array([point_dict[k] for k in point_keys])


def match(matcher, viewpoints, kds, kd1, min_matches=60):
    matches = []
    viewpoints_ = []
    for viewpoint0 in viewpoints:
        kd0 = kds[viewpoint0]
        matches01 = matcher(kd0, kd1)

        if len(matches01) < min_matches:
            continue

        matches.append(matches01)
        viewpoints_.append(viewpoint0)
    return matches, viewpoints_


def value_list(dict_, keys):
    return [dict_[k] for k in keys]


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


class VisualOdometry(object):
    def __init__(self, camera_parameters, distortion_model,
                 matcher=Matcher(enable_ransac=True,
                                 enable_homography_filter=True),
                 max_active_keyframes=8, min_matches=60):

        self.matcher = matcher
        self.min_matches = min_matches
        self.max_active_keyframes = max_active_keyframes
        self.camera_model = CameraModel(camera_parameters, distortion_model)

        self.active_viewpoints = KeyframeIndices()
        self.point_keypoint_maps = dict()

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

        viewpoint = self.active_viewpoints.get_next()
        kd = KD(self.camera_model.undistort(keypoints), descriptors)

        if len(self.active_viewpoints) == 0:
            self.init_first(viewpoint)
        elif len(self.active_viewpoints) == 1:
            self.try_init_first_two(viewpoint, kd)
        else:
            self.try_add_keyframe(viewpoint, kd)

        self.kds[viewpoint] = kd
        self.images[viewpoint] = image
        self.active_viewpoints.add_new(viewpoint)

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

    def try_init_first_two(self, viewpoint1, kd1):
        viewpoint0 = self.active_viewpoints[0]
        kd0 = self.kds[viewpoint0]

        matches01 = self.matcher(kd0, kd1)
        if len(matches01) < self.min_matches:
            raise NotEnoughInliersException("Not enough matches found")

        keypoints0, keypoints1 = kd0.keypoints, kd1.keypoints

        pose0 = Pose.identity()
        pose1 = estimate_pose_change(kd0.keypoints, kd1.keypoints, matches01)

        triangulator = Triangulation(pose0, pose1,
                                     kd0.keypoints, kd1.keypoints)
        point_array, depth_mask = triangulator.triangulate(matches01)
        matches01 = matches01[depth_mask]

        point_hashes = generate_hashes(len(point_array))
        point_dict = dict(zip(point_hashes, point_array))

        point_keypoint_map0 = init_point_keypoint_map()
        point_keypoint_map1 = init_point_keypoint_map()
        for (index0, index1), point_hash in zip(matches01, point_hashes):
            point_keypoint_map0[point_hash] = index0
            point_keypoint_map1[point_hash] = index1

        self.point_keypoint_maps[viewpoint0] = point_keypoint_map0
        self.point_keypoint_maps[viewpoint1] = point_keypoint_map1
        self.poses[viewpoint0] = pose0
        self.poses[viewpoint1] = pose1
        self.point_dict = point_dict

    def try_add_keyframe(self, viewpoint1, kd1):
        viewpoints = []
        matches = []
        for viewpoint0 in self.active_viewpoints:
            matches01 = self.matcher(self.kds[viewpoint0], kd1)
            if len(matches01) < self.min_matches:
                continue
            viewpoints.append(viewpoint0)
            matches.append(matches01)

        if len(matches) == 0:
            warnings.warn("No matches found", RuntimeWarning)
            return None

        maps = value_list(self.point_keypoint_maps, viewpoints)
        point_hashes, keypoint_indices = get_correspondences(maps, matches)
        pose1 = solve_pnp(np.array(value_list(self.point_dict, point_hashes)),
                          kd1.keypoints[keypoint_indices])

        map1s = []
        point_dicts = []
        for viewpoint0, matches01 in zip(viewpoints, matches):
            map0 = self.point_keypoint_maps[viewpoint0]
            pose0 = self.poses[viewpoint0]
            kd0 = self.kds[viewpoint0]

            # find keypoints that already have corresponding 3D points
            # if keypoint in one frame has corresponding 3D point,
            # copy it to the matched keypoint in the other frame
            indices0, indices1 = matches01[:, 0], matches01[:, 1]
            mask = copy_required(map0, indices0)
            point_hashes0 = get_point_hashes(map0, indices0[mask])

            # copy point hashes in viwepoint 0 to viewpoint 1
            # this means sharing 3D points that can be seen from viwepoint 0
            # with viewpoint 1
            map1 = init_point_keypoint_map()
            map1.update(zip(point_hashes0, indices1[mask]))

            mask = triangulation_required(map0, matches01[:, 0])
            matches01 = matches01[mask]
            t = Triangulation(pose0, pose1, kd0.keypoints, kd1.keypoints)
            point_array, depth_mask = t.triangulate(matches01)
            matches01 = matches01[depth_mask]

            point_hashes = generate_hashes(len(point_array))
            assert(len(point_hashes) == len(matches01))
            map0.update(zip(point_hashes, matches01[:, 0]))
            map1.update(zip(point_hashes, matches01[:, 1]))

            point_dict = dict(zip(point_hashes, point_array))

            map1s.append(map1)
            point_dicts.append(point_dict)

        map1 = merge_point_keypoint_maps(*map1s)
        self.point_keypoint_maps[viewpoint1] = map1
        self.point_dict = merge_dicts(self.point_dict, *point_dicts)
        self.poses[viewpoint1] = pose1
        # self.try_run_ba(self.active_viewpoints)

    def try_remove(self):
        if self.n_active_keyframes <= self.max_active_keyframes:
            return False

        viewpoint = self.active_viewpoints.remove(0)
        print("viewpoint {} has been removed".format(viewpoint))
        return True
