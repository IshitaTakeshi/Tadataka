import itertools

import numpy as np
from scipy.spatial.transform import Rotation

# TODO make this independent from cv2
import cv2

from tadataka.coordinates import local_to_world, world_to_local
from tadataka.depth import (depth_condition, warn_points_behind_cameras,
                            compute_depth_mask)
from tadataka.exceptions import NotEnoughInliersException
from tadataka.matrix import estimate_fundamental, decompose_essential
from tadataka.so3 import exp_so3, log_so3
from tadataka._triangulation import linear_triangulation


def convert_coordinates_(rotvec, t, f):
    rotvec, t = rotvec.reshape(1, -1), t.reshape(1, -1)
    rotvec, t = f(rotvec, t)
    return rotvec[0], t[0]


def convert_coordinates(pose, f):
    rotation, t = pose.rotation, pose.t
    rotvec = rotation.as_rotvec()
    rotvec, t = convert_coordinates_(rotvec, t, f)
    return Pose(Rotation.from_rotvec(rotvec), t)


class Pose(object):
    def __init__(self, rotation, translation):
        assert(isinstance(rotation, Rotation))
        self.rotation = rotation  # SciPy's Rotation object
        self.t = translation

    @property
    def R(self):
        return self.rotation.as_dcm()

    def __str__(self):
        rotvec = self.rotation.as_rotvec()
        with np.printoptions(precision=3, suppress=True):
            return "rotvec = " + str(rotvec)  + "   t = " + str(self.t)

    def world_to_local(self):
        return convert_coordinates(self, world_to_local)

    def local_to_world(self):
        return convert_coordinates(self, local_to_world)

    @staticmethod
    def identity():
        return Pose(Rotation.from_rotvec(np.zeros(3)), np.zeros(3))

    def __eq__(self, other):
        self_rotvec = self.rotation.as_rotvec()
        other_rotvec = other.rotation.as_rotvec()
        return (np.isclose(self_rotvec, other_rotvec).all() and
                np.isclose(self.t, other.t).all())


min_correspondences = 6


def calc_reprojection_threshold(keypoints, k=2.0):
    center = np.mean(keypoints, axis=0, keepdims=True)
    squared_distances = np.sum(np.power(keypoints - center, 2), axis=1)
    # rms of distances from center to keypoints
    rms = np.sqrt(np.mean(squared_distances))
    return k * rms / keypoints.shape[0]


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
    N = max(int(0.2 * len(keypoints0)), 10)
    indices = triangulation_indices(N)
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

    # we assume that keypoints are normalized
    E = estimate_fundamental(keypoints0, keypoints1)

    # R <- {R1, R2}, t <- {t1, t2} satisfy
    # K * [R | t] * homegeneous(points) = homogeneous(keypoint)
    R1A, R1B, t1a, t1b = decompose_essential(E)
    return select_valid_pose(R1A, R1B, t1a, t1b, keypoints0, keypoints1)


def estimate_pose_change(keypoints0, keypoints1):
    # estimate pose change between viewpoint 0 and 1
    # regarding viewpoint 0 as identity (world origin)
    R, t = pose_change_from_stereo(keypoints0, keypoints1)
    return Pose(Rotation.from_dcm(R), t)


def calc_relative_pose(pose0, pose1):
    """Calculate the pose change from pose0 to pose1"""

    return Pose(pose1.rotation * pose0.rotation.inv(), pose1.t - pose0.t)
