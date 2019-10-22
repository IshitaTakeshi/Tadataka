from autograd import numpy as np
from autograd import jacobian

from sba import SBA, can_run_ba

from vitamine.rigid_transform import transform
from vitamine.so3 import exp_so3
from vitamine.pose import Pose

EPSILON = 1e-16


def projection(pose, point):
    omega, t = pose[0:3], pose[3:6]
    p = transform(exp_so3(omega), t, point)
    return p[0:2] / (p[2] + EPSILON)


class Projection(object):
    def __init__(self, viewpoint_indices, point_indices):
        assert(viewpoint_indices.shape == point_indices.shape)

        self.viewpoint_indices = viewpoint_indices
        self.point_indices = point_indices

        self.n_visible = self.point_indices.shape[0]

        self.pose_jacobian = jacobian(projection, argnum=0)
        self.point_jacobian = jacobian(projection, argnum=1)

    def compute(self, poses, points):
        x_pred = np.empty((self.n_visible, 2))

        I = zip(self.viewpoint_indices, self.point_indices)
        for index, (j, i) in enumerate(I):
            x_pred[index] = projection(poses[j], points[i])
        return x_pred

    def jacobians(self, poses, points):
        A = np.zeros((self.n_visible, 2, 6))
        B = np.zeros((self.n_visible, 2, 3))
        I = zip(self.viewpoint_indices, self.point_indices)
        for index, (j, i) in enumerate(I):
            A[index] = self.pose_jacobian(poses[j], points[i])
            B[index] = self.point_jacobian(poses[j], points[i])
        return A, B


def calc_error(x_true, x_pred):
    return np.power(x_true - x_pred, 2).sum()


class LocalBundleAdjustment(object):
    def __init__(self, viewpoint_indices, point_indices, keypoints_true):
        """
        Z = zip(viewpoint_indices, pointpoint_indices)
        keypoints_true = [projection(poses[j], points[i]) for j, i in Z]
        """
        assert(len(viewpoint_indices) == keypoints_true.shape[0])
        assert(len(point_indices) == keypoints_true.shape[0])

        self.projection = Projection(viewpoint_indices, point_indices)

        self.keypoints_true = keypoints_true

        self.sba = SBA(viewpoint_indices, point_indices)

    def calc_update(self, poses, points, keypoint_pred):
        A, B = self.projection.jacobians(poses, points)
        return self.sba.compute(self.keypoints_true, keypoint_pred, A, B)

    def compute(self, initial_omegas, initial_translations, initial_points,
                n_max_iter=200, absolute_threshold=1e-2):

        poses = np.hstack((initial_omegas, initial_translations))
        points = initial_points

        keypoint_pred = self.projection.compute(poses, points)

        current_error = calc_error(self.keypoints_true, keypoint_pred)

        for _ in range(n_max_iter):
            dposes, dpoints = self.calc_update(poses, points, keypoint_pred)

            new_poses = poses + dposes
            new_points = points + dpoints

            keypoint_pred = self.projection.compute(new_poses, new_points)
            new_error = calc_error(self.keypoints_true, keypoint_pred)

            # converged or started to diverge
            if new_error > current_error:
                break

            poses = new_poses
            points = new_points

            # new_error is goood enough
            # return new_poses and new_points
            if new_error < absolute_threshold:
                break

            current_error = new_error

        omegas, translations = poses[:, 0:3], poses[:, 3:6]
        return omegas, translations, points


class IndexConverter(object):
    def __init__(self):
        self.viewpoint_indices = []
        self.point_indices = []

        self.poses = []
        self.translations = []
        self.points = []
        self.keypoints = []

        self.point_map = dict()
        self.viewpoint_map = dict()
        self.point_index = 0
        self.viewpoint_index = 0

    def add(self, viewpoint_id, point_id, pose, point, keypoint):
        if viewpoint_id not in self.viewpoint_map.keys():
            self.viewpoint_map[viewpoint_id] = self.viewpoint_index
            self.viewpoint_index += 1
            self.poses.append(pose)

        if point_id not in self.point_map.keys():
            self.point_map[point_id] = self.point_index
            self.point_index += 1
            self.points.append(point)

        self.viewpoint_indices.append(self.viewpoint_map[viewpoint_id])
        self.point_indices.append(self.point_map[point_id])
        self.keypoints.append(keypoint)

    def export_projection(self):
        return (np.array(self.viewpoint_indices),
                np.array(self.point_indices),
                np.array(self.keypoints))

    def export_pose_points(self):
        return self.poses, self.points


def get_converter(index_map, poses, points, keypoints_list, viewpoints):
    converter = IndexConverter()
    for viewpoint, pose, keypoints in zip(viewpoints, poses, keypoints_list):
        M = index_map[viewpoint]
        assert(len(M.keys()) == len(set(M.values())))
        for keypoint_index, point_index in index_map[viewpoint].items():
            converter.add(viewpoint, point_index, pose,
                          points[point_index], keypoints[keypoint_index])
    return converter


def try_run_ba_(viewpoint_indices, point_indices,
                poses, points, keypoints_true):
    """
    points:
    """

    ba = LocalBundleAdjustment(viewpoint_indices, point_indices,
                               keypoints_true)

    omegas = np.array([p.omega for p in poses])
    ts = np.array([p.t for p in poses])

    omegas, ts, points = ba.compute(omegas, ts, points)

    poses = [Pose(omega, t) for omega, t in zip(omegas, ts)]
    return poses, points


def test_unique(viewpoint_indices, point_indices):
    A = np.vstack((viewpoint_indices, point_indices))
    assert(np.unique(A, axis=1).shape[1] == A.shape[1])


def try_run_ba(index_map, points, poses, keypoints_list, viewpoints):
    assert(len(poses) == len(keypoints_list) == len(viewpoints))

    converter = get_converter(index_map, poses, points,
                              keypoints_list, viewpoints)

    viewpoint_indices, point_indices, keypoints_true =\
        converter.export_projection()
    local_poses, local_points = converter.export_pose_points()

    test_unique(viewpoint_indices, point_indices)

    if not can_run_ba(n_viewpoints=len(local_poses),
                      n_points=len(local_points),
                      n_visible=len(keypoints_true),
                      n_pose_params=6, n_point_params=3):
        return local_poses, local_points
    return try_run_ba_(viewpoint_indices, point_indices,
                       local_poses, local_points, keypoints_true)
