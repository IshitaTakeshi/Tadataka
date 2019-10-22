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
