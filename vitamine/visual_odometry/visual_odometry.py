from autograd import numpy as np

from vitamine.exceptions import NotEnoughInliersException, print_error
from vitamine.keypoints import extract_keypoints, Matcher
from vitamine.keypoints import KeypointDescriptor as KD
from vitamine.camera_distortion import CameraModel
from vitamine.point_keypoint_map import (
    init_point_keypoint_map, copy_existing_points, correspondences,
    triangulation_required
)
from vitamine.utils import merge_dicts
from vitamine.pose import Pose, solve_pnp, estimate_pose_change
from vitamine.triangulation import Triangulation
from vitamine.keyframe_index import KeyframeIndices
from vitamine.random import random_bytes
from vitamine.so3 import rodrigues

from skimage.color import rgb2gray
from vitamine.local_ba import try_run_ba


def triangulation(pose0, pose1, keypoints0, keypoints1, matches01):
    triangulator = Triangulation(pose0, pose1, keypoints0, keypoints1)
    point_array, depth_mask = triangulator.triangulate(matches01)
    return point_array, matches01[depth_mask]


def generate_hashes(n_hashes, n_bytes=18):
    return [random_bytes(n_bytes) for i in range(n_hashes)]


def init_from_two_views(matcher, kd0, kd1, min_matches=60):
    matches01 = matcher(kd0, kd1)
    if len(matches01) < min_matches:
        raise NotEnoughInliersException("Not enough matches found")

    pose0 = Pose.identity()
    pose1 = estimate_pose_change(kd0.keypoints, kd1.keypoints, matches01)

    triangulator = Triangulation(pose0, pose1, kd0.keypoints, kd1.keypoints)
    point_array, depth_mask = triangulator.triangulate(matches01)
    matches01 = matches01[depth_mask]

    point_hashes = generate_hashes(len(point_array))
    point_dict = dict(zip(point_hashes, point_array))

    point_keypoint_map0 = init_point_keypoint_map()
    point_keypoint_map1 = init_point_keypoint_map()

    for (index0, index1), point_hash in zip(matches01, point_hashes):
        point_keypoint_map0[point_hash] = index0
        point_keypoint_map1[point_hash] = index1

    return point_keypoint_map0, point_keypoint_map1, pose0, pose1, point_dict


def point_array(point_dict, point_keys):
    return np.array([point_dict[k] for k in point_keys])


def try_run_ba(self, viewpoints):
    poses = [self.poses[v] for v in viewpoints]
    keypoints = [self.kds[v].keypoints for v in viewpoints]

    try:
        poses, points, point_indices = try_run_ba(
            self.point_manager.index_map,
            self.point_manager.points,
            poses, keypoints, viewpoints
        )
    except ValueError as e:
        print_error(e)
        return

    assert(len(point_indices) == len(points))

    for j, pose in zip(viewpoints, poses):
        self.poses[j] = pose

    for i, point in zip(point_indices, points):
        self.point_manager.overwrite(i, point)


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


def update(matches01, point_keypoint_map0, point_keypoint_map1,
           pose0, pose1, kd0, kd1):

    point_keypoint_map0, point_keypoint_map1, matches01 = associate_existing(
        point_keypoint_map0, point_keypoint_map1, matches01)

    point_array, matches01 = triangulation(pose0, pose1,
                                           kd0.keypoints, kd1.keypoints,
                                           matches01)

    point_hashes = generate_hashes(len(point_array))
    point_dict = dict(zip(point_hashes, point_array))

    for (index0, index1), point_hash in zip(matches01, point_hashes):
        point_keypoint_map0[point_hash] = index0
        point_keypoint_map1[point_hash] = index1

    return point_keypoint_map0, point_keypoint_map1, point_dict


class VisualOdometry(object):
    def __init__(self, camera_parameters, distortion_model,
                 matcher=Matcher(enable_ransac=True,
                                 enable_homography_filter=True),
                 max_active_keyframes=8):

        self.matcher = matcher
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

        self.try_add_keyframe(viewpoint, kd, image)

        self.active_viewpoints.add_new(viewpoint)
        self.kds[viewpoint] = kd
        self.images[viewpoint] = image
        return viewpoint

    def try_add_keyframe(self, viewpoint1, kd1, image1):
        min_matches = 60
        if len(self.active_viewpoints) == 0:
            self.poses[viewpoint1] = Pose.identity()
            return

        if len(self.active_viewpoints) == 1:
            viewpoint0 = self.active_viewpoints[0]
            kd0 = self.kds[viewpoint0]

            matches01 = self.matcher(kd0, kd1)
            if len(matches01) < min_matches:
                raise NotEnoughInliersException("Not enough matches found")

            pose0 = Pose.identity()
            pose1 = estimate_pose_change(kd0.keypoints, kd1.keypoints, matches01)

            triangulator = Triangulation(pose0, pose1, kd0.keypoints, kd1.keypoints)
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
            self.point_dict = point_dict
            self.poses[viewpoint0] = pose0
            self.poses[viewpoint1] = pose1
            return

        viewpoints = []
        matches = []
        for viewpoint0 in self.active_viewpoints:
            matches01 = self.matcher(self.kds[viewpoint0], kd1)
            if len(matches01) < min_matches:
                continue
            viewpoints.append(viewpoint0)
            matches.append(matches01)

        if len(matches) == 0:
            raise NotEnoughInliersException("No matches found")
            return None

        maps = value_list(self.point_keypoint_maps, viewpoints)
        point_keys, keypoint_indices = correspondences(maps, matches)
        pose1 = solve_pnp(np.array(value_list(self.point_dict, point_keys)),
                          kd1.keypoints[keypoint_indices])

        point_dict = dict()
        point_keypoint_map1 = init_point_keypoint_map()
        for viewpoint0, matches01 in zip(viewpoints, matches):
            point_keypoint_map0 = self.point_keypoint_maps[viewpoint0]
            pose0 = self.poses[viewpoint0]
            kd0 = self.kds[viewpoint0]

            mask = triangulation_required(point_keypoint_map0, point_keypoint_map1, matches01)

            point_keypoint_map0, point_keypoint_map1 = copy_existing_points(
                point_keypoint_map0, point_keypoint_map1, matches01[~mask]
            )

            matches01 = matches01[mask]
            triangulator = Triangulation(pose0, pose1, kd0.keypoints, kd1.keypoints)
            point_array, depth_mask = triangulator.triangulate(matches01)

            point_hashes = generate_hashes(len(point_array))

            assert(len(matches01[depth_mask]) == len(point_hashes))
            for (index0, index1), point_hash in zip(matches01[depth_mask], point_hashes):
                point_keypoint_map0[point_hash] = index0
                point_keypoint_map1[point_hash] = index1

            for point_hash, point in zip(point_hashes, point_array):
                point_dict[point_hash] = point

        self.point_keypoint_maps[viewpoint1] = point_keypoint_map1
        self.point_dict = merge_dicts(self.point_dict, point_dict)
        self.poses[viewpoint1] = pose1
        # self.try_run_ba(self.active_viewpoints)

    def try_remove(self):
        if self.n_active_keyframes <= self.max_active_keyframes:
            return False

        viewpoint = self.active_viewpoints.remove(0)
        print("viewpoint {} has been removed".format(viewpoint))
        return True
