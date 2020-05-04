import itertools

import numpy as np
from scipy.spatial.transform import Rotation

# TODO make this independent from cv2
import cv2

from tadataka.depth import (depth_condition, warn_points_behind_cameras,
                            compute_depth_mask)
from tadataka.exceptions import NotEnoughInliersException
from tadataka.matrix import (estimate_fundamental, decompose_essential,
                             motion_matrix)
from tadataka.so3 import exp_so3, log_so3
from tadataka.se3 import exp_se3_t_
from tadataka.triangulation import linear_triangulation


def check_type_(pose1, pose2):
    if type(pose1) == type(pose2):
        return

    name1 = type(pose1).__name__
    name2 = type(pose2).__name__
    raise ValueError(f"Types do not match: {name1} and {name2}")


class Pose(object):
    def __init__(self, rotation, translation):
        assert(isinstance(rotation, Rotation))
        self.rotation = rotation  # SciPy's Rotation object
        self.t = translation

    @property
    def R(self):
        return self.rotation.as_matrix()

    @property
    def T(self):
        return motion_matrix(self.R, self.t)

    def __str__(self):
        rotvec = self.rotation.as_rotvec()
        sr = ' '.join(["{: .3f}".format(v) for v in rotvec])
        st = ' '.join(["{: .3f}".format(v) for v in self.t])
        return "rotvec = [ " + sr + " ]  t = [ " + st + " ]"

    @classmethod
    def identity(PoseClass):
        return PoseClass(Rotation.from_rotvec(np.zeros(3)), np.zeros(3))

    @classmethod
    def from_se3(PoseClass, xi):
        rotvec = xi[3:]
        return PoseClass(Rotation.from_rotvec(rotvec), exp_se3_t_(xi))

    def inv(self):
        PoseClass = type(self)
        return PoseClass(*convert_coordinate(self.rotation, self.t))

    def __mul__(self, other):
        check_type_(self, other)
        PoseClass = type(self)
        return PoseClass(self.rotation * other.rotation,
                         np.dot(self.R, other.t) + self.t)

    def __eq__(self, other):
        check_type_(self, other)
        self_rotvec = self.rotation.as_rotvec()
        other_rotvec = other.rotation.as_rotvec()
        return (np.isclose(self_rotvec, other_rotvec).all() and
                np.isclose(self.t, other.t).all())


def convert_coordinate(rotation, t):
    inv_rotation = rotation.inv()
    return inv_rotation, -np.dot(inv_rotation.as_matrix(), t)


def calc_reprojection_threshold(keypoints, k=2.0):
    center = np.mean(keypoints, axis=0, keepdims=True)
    squared_distances = np.sum(np.power(keypoints - center, 2), axis=1)
    # rms of distances from center to keypoints
    rms = np.sqrt(np.mean(squared_distances))
    return k * rms / keypoints.shape[0]


min_correspondences = 6


def solve_pnp(points, keypoints):
    assert(points.shape[0] == keypoints.shape[0])

    if keypoints.shape[0] < min_correspondences:
        raise NotEnoughInliersException("No sufficient correspondences")

    t = calc_reprojection_threshold(keypoints, k=3.0)
    retval, omega, t, inliers = cv2.solvePnPRansac(
        points.astype(np.float64),
        keypoints.astype(np.float64),
        np.identity(3), np.zeros(4),
        reprojectionError=t,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not retval:
        raise RuntimeError("Pose estimation failed")

    if len(inliers.flatten()) == 0:
        raise NotEnoughInliersException("No inliers found")

    return Pose(Rotation.from_rotvec(omega.flatten()), t.flatten())


# We triangulate only subset of keypoints to determine valid
# (rotation, translation) pair
def n_triangulated(n_keypoints, triangulation_ratio=0.2, n_min_triangulation=40):
    n = int(n_keypoints * triangulation_ratio)
    # at least use 'n_min_triangulation' points
    k = max(n, n_min_triangulation)
    # make the return value not exceed the number of keypoints
    return min(n_keypoints, k)


def triangulation_indices(n_keypoints):
    N = n_triangulated(n_keypoints)
    indices = np.arange(0, n_keypoints)
    np.random.shuffle(indices)
    return indices[:N]


def select_valid_pose(R1A, R1B, t1a, t1b, keypoints0, keypoints1):
    R0, t0 = np.identity(3), np.zeros(3)

    n_max_valid_depth = -1
    argmax_R, argmax_t, argmax_depth_mask = None, None, None

    # not necessary to triangulate all points to validate depths
    indices = triangulation_indices(min(100, len(keypoints0)))
    keypoints = np.stack((keypoints0[indices], keypoints1[indices]))
    for i, (R_, t_) in enumerate(itertools.product((R1A, R1B), (t1a, t1b))):
        _, depths = linear_triangulation(
            np.array([R0, R_]),
            np.array([t0, t_]),
            keypoints
        )

        depth_mask = compute_depth_mask(depths)
        n_valid_depth = np.sum(depth_mask)

        # only 1 pair (R, t) among the candidates has to be
        # the correct pair
        if n_valid_depth > n_max_valid_depth:
            n_max_valid_depth = n_valid_depth
            argmax_R, argmax_t, argmax_depth_mask = R_, t_, depth_mask

    if not depth_condition(argmax_depth_mask):
        warn_points_behind_cameras()

    return argmax_R, argmax_t


def pose_change_from_stereo(keypoints0, keypoints1):
    """Estimate camera pose change between two viewpoints"""

    assert(keypoints0.shape == keypoints1.shape)

    # we assume that the keypoints are normalized
    E = estimate_fundamental(keypoints0, keypoints1)

    R10A, R10B, t10a, t10b = decompose_essential(E)
    return select_valid_pose(R10A, R10B, t10a, t10b, keypoints0, keypoints1)


def estimate_pose_change(keypoints0, keypoints1):
    """
    Estimate pose s.t. X1 = pose.R * X0 + pose.t
    where keypoints0 = pi(X0), keypoints1 = pi(X1)
    """
    R10, t10 = pose_change_from_stereo(keypoints0, keypoints1)
    return Pose(Rotation.from_matrix(R10), t10)
