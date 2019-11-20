import warnings

from autograd import numpy as np
from autograd import jacobian

from sba import SBA, can_run_ba

from tadataka.rigid_transform import transform
from tadataka.so3 import exp_so3
from tadataka.pose import Pose


EPSILON = 1e-16


def projection(pose, point):
    omega, t = pose[0:3], pose[3:6]
    p = transform(exp_so3(omega), t, point)
    return p[0:2] / (p[2] + EPSILON)


def create_jacobian():
    index = 0
    for i, j in itertools.product(range(n_points), range(n_viewpoints)):
        if not mask[i, j]:
            continue

        viewpoint_indices[index] = j
        point_indices[index] = i

        row = index * 2

        col = j * n_pose_params
        JA[row:row+2, col:col+n_pose_params] = A[index]

        col = i * n_point_params
        JB[row:row+2, col:col+n_point_params] = B[index]

        index += 1


class Projection(object):
    def __init__(self, viewpoint_indices, point_indices):
        assert(len(viewpoint_indices) == len(point_indices))

        self.viewpoint_indices = viewpoint_indices
        self.point_indices = point_indices

        self.n_visible = len(self.point_indices)

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
        self.point_ids = []

    def add(self, viewpoint_id, point_id, pose, point, keypoint):
        if viewpoint_id not in self.viewpoint_map.keys():
            self.viewpoint_map[viewpoint_id] = self.viewpoint_index
            self.viewpoint_index += 1
            self.poses.append(pose)

        if point_id not in self.point_map.keys():
            self.point_map[point_id] = self.point_index
            self.point_index += 1
            self.point_ids.append(point_id)
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


def run_ba(viewpoint_indices, point_indices,
                poses, points, keypoints_true):
    ba = LocalBundleAdjustment(viewpoint_indices, point_indices,
                               keypoints_true)

    omegas = np.array([p.omega for p in poses])
    ts = np.array([p.t for p in poses])

    omegas, ts, points = ba.compute(omegas, ts, points,
                                    absolute_error_threshold=1e-9,
                                    relative_error_threshold=0.20)

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
