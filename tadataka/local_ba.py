import warnings

import numpy as np

from sba import SBA, can_run_ba

from tadataka.rigid_transform import transform
from tadataka.pose import Pose
from tadataka.so3_codegen import projection, pose_jacobian, point_jacobian


class Projection(object):
    def __init__(self, viewpoint_indices, point_indices):
        assert(len(viewpoint_indices) == len(point_indices))

        self.viewpoint_indices = viewpoint_indices
        self.point_indices = point_indices

        self.n_visible = len(self.point_indices)

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
            A[index] = pose_jacobian(poses[j], points[i])
            B[index] = point_jacobian(poses[j], points[i])
        return A, B


def calc_relative_error(current_error, new_error):
    return np.abs((current_error - new_error) / new_error)


def calc_errors(x_true, x_pred):
    return np.sum(np.power(x_true - x_pred, 2), axis=1)


def calc_error(x_true, x_pred):
    return np.mean(calc_errors(x_true, x_pred))


def update_weights(robustifier, x_true, x_pred, weights):
    E = calc_errors(x_true, x_pred, weights)
    I = np.identity(2)
    return np.array([I * w for w in robustifier.weights(E)])


class LocalBundleAdjustment(object):
    def __init__(self, viewpoint_indices, point_indices, x_true):
        """
        Z = zip(viewpoint_indices, pointpoint_indices)
        x_true = [projection(poses[j], points[i]) for j, i in Z]
        """
        assert(len(viewpoint_indices) == x_true.shape[0])
        assert(len(point_indices) == x_true.shape[0])

        self.projection = Projection(viewpoint_indices, point_indices)
        self.x_true = x_true

        self.sba = SBA(viewpoint_indices, point_indices)

    def calc_update(self, poses, points, mu):
        x_pred = self.projection.compute(poses, points)
        A, B = self.projection.jacobians(poses, points)
        return self.sba.compute(self.x_true, x_pred, A, B, weights=None, mu=mu)

    def calc_error(self, poses, points):
        x_pred = self.projection.compute(poses, points)
        return calc_error(self.x_true, x_pred)

    def calc_new_error(self, poses, points, mu):
        dposes, dpoints = self.calc_update(poses, points, mu)
        error = self.calc_error(poses + dposes, points + dpoints)
        return dposes, dpoints, error

    def lm_update(self, poses, points, mu, nu):
        error0 = self.calc_error(poses, points)

        new_mu = mu / nu
        dposes, dpoints, error = self.calc_new_error(poses, points, new_mu)
        if error < error0:
            return poses + dposes, points + dpoints, new_mu, error

        new_mu = mu
        dposes, dpoints, error = self.calc_new_error(poses, points, new_mu)
        if error < error0:
            return poses + dposes, points + dpoints, new_mu, error

        error = np.inf
        new_mu = mu
        while error > error0:
            new_mu = new_mu * nu
            dposes, dpoints, error = self.calc_new_error(poses, points, new_mu)
        return poses + dposes, points + dpoints, new_mu, error

    def compute(self, initial_omegas, initial_translations, initial_points,
                max_iter=200, initial_mu=1.0, nu=100.0,
                absolute_error_threshold=1e-8, relative_error_threshold=1e-6):

        poses = np.hstack((initial_omegas, initial_translations))
        points = initial_points

        mu = initial_mu
        current_error = self.calc_error(poses, points)
        for iter_ in range(max_iter):
            poses, points, mu, new_error = self.lm_update(poses, points, mu, nu)

            relative_error = calc_relative_error(current_error, new_error)

            print(f"absolute_error[{iter_}] = {new_error}")
            print(f"relative_error[{iter_}] = {relative_error}")

            if new_error < absolute_error_threshold:
                break

            if relative_error < relative_error_threshold:
                break

            current_error = new_error

        omegas, translations = poses[:, 0:3], poses[:, 3:6]
        return omegas, translations, points


def run_ba(viewpoint_indices, point_indices,
                poses, points, keypoints_true):
    ba = LocalBundleAdjustment(viewpoint_indices, point_indices,
                               keypoints_true)

    omegas = np.array([p.omega for p in poses])
    ts = np.array([p.t for p in poses])

    omegas, ts, points = ba.compute(omegas, ts, points,
                                    absolute_error_threshold=1e-9,
                                    relative_error_threshold=0.10)

    poses = [Pose(omega, t) for omega, t in zip(omegas, ts)]
    return poses, points


def test_unique(viewpoint_indices, point_indices):
    A = np.vstack((viewpoint_indices, point_indices))
    assert(np.unique(A, axis=1).shape[1] == A.shape[1])


def try_run_ba(viewpoint_indices, point_indices,
               poses, points, keypoints_true):
    assert(len(viewpoint_indices) == len(point_indices))
    assert(len(set(viewpoint_indices)) == len(poses))
    assert(len(set(point_indices)) == len(points))

    test_unique(viewpoint_indices, point_indices)

    if not can_run_ba(n_viewpoints=len(poses),
                      n_points=len(points),
                      n_visible=len(keypoints_true),
                      n_pose_params=6, n_point_params=3):
        warnings.warn("Arguments are not satisfying condition to run BA",
                      RuntimeWarning)
        return poses, points
        # raise ValueError("Arguments are not satisfying condition to run BA")

    return run_ba(viewpoint_indices, point_indices,
                  poses, points, keypoints_true)
