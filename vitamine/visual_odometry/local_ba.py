from autograd import numpy as np
from autograd import jacobian
# from julia.SBA import Indices, sba

from vitamine.bundle_adjustment.mask import keypoint_mask
from vitamine.rigid_transform import transform_each
from vitamine.so3 import rodrigues
from vitamine.projection import pi


EPSILON = 1e-16


def reverse_axes_3d(array):
    return np.swapaxes(np.swapaxes(array, 0, 2), 0, 1)


class Projection(object):
    def __init__(self, camera_parameters):
        self.K = camera_parameters.matrix

    def compute(self, pose, point):
        omega, t = pose[0:3], pose[3:6]
        R = rodrigues(omega.reshape(1, -1))[0]
        p = np.dot(R, point) + t
        p = np.dot(self.K, p)
        return p[0:2] / (p[2] + EPSILON)


class KeypointPrediction(object):
    def __init__(self, viewpoint_indices, point_indices, projection):
        assert(viewpoint_indices.shape == point_indices.shape)
        self.projection = projection

        self.viewpoint_indices = viewpoint_indices
        self.point_indices = point_indices

        self.N = self.point_indices.shape[0]

        self.pose_jacobian = jacobian(self.projection.compute, argnum=0)
        self.point_jacobian = jacobian(self.projection.compute, argnum=1)

    def compute(self, poses, points):
        x_pred = np.empty((self.N, 2))

        I = zip(self.viewpoint_indices, self.point_indices)
        for index, (j, i) in enumerate(I):
            x_pred[index] = self.projection.compute(poses[j], points[i])
        return x_pred

    def jacobians(self, poses, points):
        N = self.viewpoint_indices.shape[0]

        A = np.zeros((N, 2, 6))
        B = np.zeros((N, 2, 3))
        I = zip(self.viewpoint_indices, self.point_indices)
        for index, (j, i) in enumerate(I):
            A[index] = self.pose_jacobian(poses[j], points[i])
            B[index] = self.point_jacobian(poses[j], points[i])
        return A, B


def calc_error(x_true, x_pred):
    return np.power(x_true - x_pred, 2).sum()


def check_params(poses, points, keypoints):
    n_poses, n_pose_params = poses.shape
    n_points, n_point_params = points.shape

    n_visible_keypoints = keypoints.shape[0]
    n_rows = 2 * n_visible_keypoints
    n_cols = n_poses * n_pose_params + n_points * n_point_params
    assert(n_rows >= n_cols)


class LocalBundleAdjustment(object):
    def __init__(self, viewpoint_indices, point_indices, keypoints_true,
                 camera_parameters):
        """
        I = zip(viewpoint_indices, pointpoint_indices)
        keypoints_true = [projection(poses[j], points[i]) for j, i in I]
        """
        assert(len(viewpoint_indices) == len(point_indices) == keypoints_true.shape[0])

        self.prediction = KeypointPrediction(viewpoint_indices, point_indices,
                                             Projection(camera_parameters))

        self.keypoints_true = keypoints_true

        # increment every index as Julia start them from 1
        self.indices = Indices(viewpoint_indices + 1, point_indices + 1)

    def calc_update(self, poses, points, keypoint_pred):
        assert(keypoint_pred.shape == self.keypoints_true.shape)
        check_params(poses, points, keypoint_pred)

        A, B = self.prediction.jacobians(poses, points)
        dposes, dpoints = sba(self.indices,
                              np.swapaxes(self.keypoints_true, 0, 1),
                              np.swapaxes(keypoint_pred, 0, 1),
                              reverse_axes_3d(A), reverse_axes_3d(B))
        assert(not np.isnan(dposes).all())
        assert(not np.isnan(dpoints).all())
        return dposes.T, dpoints.T

    def compute(self, initial_omegas, initial_translations, initial_points,
                n_max_iter=200, absolute_threshold=1e-2):

        poses = np.hstack((initial_omegas, initial_translations))
        points = initial_points

        keypoint_pred = self.prediction.compute(poses, points)
        current_error = calc_error(self.keypoints_true, keypoint_pred)

        for iter_ in range(n_max_iter):
            dposes, dpoints = self.calc_update(poses, points, keypoint_pred)

            new_poses = poses + dposes
            new_points = points + dpoints

            keypoint_pred = self.prediction.compute(new_poses, new_points)
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
