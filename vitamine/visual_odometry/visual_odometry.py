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
                # print_error(e)
                continue

            keypoint = keypoints[index1]

            points.append(point)
            keypoints_.append(keypoint)
    return np.array(points), np.array(keypoints_)


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


def try_init_first_two(point_manager, matcher, kd0, kd1,
                       viewpoint0, viewpoint1, min_matches=8):
    matches01 = matcher(kd0, kd1)

    if len(matches01) < min_matches:
        raise NotEnoughInliersException("Not enough matches found")

    pose0, pose1 = point_manager.initialize(
        kd0.keypoints, kd1.keypoints, matches01,
        viewpoint0, viewpoint1
    )
    return point_manager, pose0, pose1


def try_add_more(point_manager, matcher,
                 poses, kds, viewpoints,
                 kd1, viewpoint1):
    matches, viewpoints = match(matcher, viewpoints, kds, kd1)
    assert(len(matches) == len(viewpoints))

    if len(matches) == 0:
        raise NotEnoughInliersException("No matches found")

    points, keypoints = accumulate_correspondences(
        point_manager, kd1.keypoints,
        matches, viewpoints
    )

    pose1 = solve_pnp(points, keypoints)

    # if not self.pose_condition(poses[-1], pose1):
    #     # if pose1 is too close from the latest active pose
    #     return None

    for viewpoint0, matches01 in zip(viewpoints, matches):
        kd0 = kds[viewpoint0]
        pose0 = poses[viewpoint0]
        point_manager.triangulate(
            pose0, pose1, kd0.keypoints, kd1.keypoints, matches01,
            viewpoint0, viewpoint1
        )

    return point_manager, pose1


class VisualOdometry(object):
    def __init__(self, camera_parameters, distortion_model,
                 matcher=Matcher(enable_ransac=True,
                                 enable_homography_filter=True),
                 max_active_keyframes=8):
        self.matcher = matcher
        self.max_active_keyframes = max_active_keyframes
        self.camera_model = CameraModel(camera_parameters, distortion_model)
        self.active_viewpoints = KeyframeIndices()
        self.point_manager = PointManager()
        self.kds = dict()
        self.poses = dict()
        self.images = dict()

        self.kds_ = dict()

    def export_points(self):
        return self.point_manager.export_points()

    def export_poses(self):
        poses = [self.poses[v] for v in sorted(self.poses.keys())]
        return [[pose.omega, pose.t] for pose in poses]

    @property
    def n_active_keyframes(self):
        return len(self.active_viewpoints)

    def add(self, image, min_keypoints=8):
        keypoints, descriptors = extract_keypoints(image)

        if len(keypoints) <= min_keypoints:
            print_error("Keypoints not sufficient")
            return -1

        viewpoint = self.active_viewpoints.get_next()
        kd = KD(self.camera_model.undistort(keypoints), descriptors)

        from vitamine.plot.debug import plot_matches
        from matplotlib import pyplot as plt
        for v in self.active_viewpoints:
            kdv = self.kds_[v]
            imagev = self.images[v]
            matches = self.matcher(kd, kdv)
            print("n_matches =", len(matches))
            plot_matches(image, imagev,
                         keypoints, kdv.keypoints,
                         matches)
        self.kds_[viewpoint] = KD(keypoints, descriptors)
        plt.show()

        pose = self.try_add_keyframe(viewpoint, kd, image)
        if pose is None:
            print_error("Pose estimation failed")
            return -1

        self.active_viewpoints.add_new(viewpoint)
        self.kds[viewpoint] = kd
        self.poses[viewpoint] = pose
        self.images[viewpoint] = image
        return viewpoint

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

        assert(len(point_indices) == len(points))

        for j, pose in zip(viewpoints, poses):
            self.poses[j] = pose

        for i, point in zip(point_indices, points):
            self.point_manager.overwrite(i, point)

    def try_add_keyframe(self, new_viewpoint, new_kd, new_image):
        if self.n_active_keyframes == 0:
            return Pose.identity()

        if self.n_active_keyframes == 1:
            viewpoint0 = self.active_viewpoints[-1]
            try:
                self.point_manager, _, pose1 = try_init_first_two(
                    point_manager=self.point_manager, matcher=self.matcher,
                    kd0=self.kds[0], kd1=new_kd,
                    viewpoint0=viewpoint0, viewpoint1=new_viewpoint
                )
            except NotEnoughInliersException as e:
                print_error(e)
                return None

            return pose1

        active_viewpoints = self.active_viewpoints
        active_kds = {v: self.kds[v] for v in active_viewpoints}
        active_poses = {v: self.poses[v] for v in active_viewpoints}
        try:
            self.point_manager, pose1 = try_add_more(
                self.point_manager, self.matcher,
                active_poses, active_kds, active_viewpoints,
                new_kd, new_viewpoint,
            )
        except NotEnoughInliersException as e:
            print_error(e)
            return None

        self.try_run_ba(self.active_viewpoints)

        return pose1

    def try_remove(self):
        if self.n_active_keyframes <= self.max_active_keyframes:
            return False

        viewpoint = self.active_viewpoints.remove(0)
        print("viewpoint {} has been removed".format(viewpoint))
        return True
