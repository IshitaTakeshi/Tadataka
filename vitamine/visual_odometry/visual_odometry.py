from autograd import numpy as np

from vitamine.exceptions import NotEnoughInliersException, print_error
from vitamine.keypoints import extract_keypoints, Matcher
from vitamine.keypoints import KeypointDescriptor as KD
from vitamine.camera_distortion import CameraModel
from vitamine.points import PointManager
from vitamine.pose import Pose, solve_pnp, estimate_pose_change
from vitamine.triangulation import Triangulation
from vitamine.keyframe_index import KeyframeIndices
from skimage.color import rgb2gray
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
                # keypoint corresponding to 'index0' in 'viewpoint' is not
                # triangulated yet
                continue

            points.append(point)
            keypoints_.append(keypoints[index1])

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


def triangulation(point_manager, matches, viewpoints, viewpoint1, kds, kd1, poses, pose1):
    for viewpoint0, matches01 in zip(viewpoints, matches):
        kd0 = kds[viewpoint0]
        pose0 = poses[viewpoint0]
        triangulator = Triangulation(pose0, pose1,
                                     kd0.keypoints, kd1.keypoints)
        point_manager.triangulate(triangulator, matches01,
                                  viewpoint0, viewpoint1)
    return point_manager


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
        keypoints, descriptors = extract_keypoints(rgb2gray(image))

        if len(keypoints) <= min_keypoints:
            print_error("Keypoints not sufficient")
            return -1

        viewpoint = self.active_viewpoints.get_next()
        kd = KD(self.camera_model.undistort(keypoints), descriptors)

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
            return

        assert(len(point_indices) == len(points))

        for j, pose in zip(viewpoints, poses):
            self.poses[j] = pose

        for i, point in zip(point_indices, points):
            self.point_manager.overwrite(i, point)

    def try_add_keyframe(self, viewpoint1, kd1, image1):
        min_matches = 60
        if len(self.active_viewpoints) == 0:
            return Pose.identity()

        if len(self.active_viewpoints) == 1:
            viewpoint0 = self.active_viewpoints[0]
            pose0 = self.poses[viewpoint0]
            kd0 = self.kds[viewpoint0]

            matches01 = self.matcher(kd0, kd1)
            pose1 = estimate_pose_change(kd0.keypoints, kd1.keypoints, matches01)

            if len(matches01) < min_matches:
                raise NotEnoughInliersException("Not enough matches found")

            triangulator = Triangulation(pose0, pose1,
                                         kd0.keypoints, kd1.keypoints)
            for index0, index1 in matches01:
                point = triangulator.triangulate(index0, index1)
                self.point_manager.add_point(point, viewpoint0, viewpoint1,
                                             index0, index1)

            return pose1

        matches, viewpoints = match(self.matcher, self.active_viewpoints, self.kds, kd1)
        assert(len(matches) == len(viewpoints))

        if len(matches) == 0:
            raise NotEnoughInliersException("No matches found")
            return None

        points, keypoints = accumulate_correspondences(
            self.point_manager, kd1.keypoints,
            matches, viewpoints
        )
        pose1 = solve_pnp(points, keypoints)

        self.point_manager = triangulation(self.point_manager, matches,
                                           viewpoints, viewpoint1,
                                           self.kds, kd1, self.poses, pose1)

        self.try_run_ba(self.active_viewpoints)

        return pose1

    def try_remove(self):
        if self.n_active_keyframes <= self.max_active_keyframes:
            return False

        viewpoint = self.active_viewpoints.remove(0)
        print("viewpoint {} has been removed".format(viewpoint))
        return True
