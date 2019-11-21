import itertools

import numpy as np

# TODO make this independent from cv2
import cv2

from tadataka.exceptions import NotEnoughInliersException
from tadataka.matrix import estimate_fundamental, decompose_essential
from tadataka.so3 import exp_so3, log_so3
from tadataka._triangulation import triangulation_
from tadataka.depth import depth_condition, warn_points_behind_cameras


class Pose(object):
    def __init__(self, R_or_omega, t):
        if np.ndim(R_or_omega) == 1:
            self.omega = R_or_omega
        elif np.ndim(R_or_omega) == 2:
            self.omega = log_so3(R_or_omega)

        self.t = t

    @property
    def R(self):
        return exp_so3(self.omega)

    def __str__(self):
        with np.printoptions(precision=3, suppress=True):
            return "omega = " + str(self.omega)  + "   t = " + str(self.t)

    @staticmethod
    def identity():
        return Pose(np.zeros(3), np.zeros(3))

    def __eq__(self, other):
        return (np.isclose(self.omega, other.omega).all() and
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

    print("keypoints.shape", keypoints.shape)
    t = calc_reprojection_threshold(keypoints, k=3.0)
    print("reprojectionError", t)
    retval, omega, t, inliers = cv2.solvePnPRansac(
        points.astype(np.float64),
        keypoints.astype(np.float64),
        np.identity(3), np.zeros(4),
        reprojectionError=t,
        flags=cv2.SOLVEPNP_EPNP
    )
    print("retval: {}".format(retval))
    print("inlier ratio")
    print(len(inliers.flatten()) / points.shape[0])

    if len(inliers.flatten()) == 0:
        raise NotEnoughInliersException("No inliers found")

    return Pose(omega.flatten(), t.flatten())


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


def select_valid_pose(R1, R2, t1, t2, keypoints0, keypoints1):
    R0, t0 = np.identity(3), np.zeros(3)

    n_max_valid_depth = -1
    argmax_R, argmax_t, argmax_depth_mask = None, None, None

    # not necessary to triangulate all points to validate depths
    indices = triangulation_indices(len(keypoints0))
    for i, (R_, t_) in enumerate(itertools.product((R1, R2), (t1, t2))):
        _, depth_mask = triangulation_(
            R0, R_, t0, t_, keypoints0[indices], keypoints1[indices]
        )
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
    R1, R2, t1, t2 = decompose_essential(E)
    return select_valid_pose(R1, R2, t1, t2, keypoints0, keypoints1)


def estimate_pose_change(keypoints0, keypoints1, matches01):
    # estimate pose change between viewpoint 0 and 1
    # regarding viewpoint 0 as identity (world origin)
    R, t = pose_change_from_stereo(
        keypoints0[matches01[:, 0]],
        keypoints1[matches01[:, 1]]
    )

    return Pose(R, t)
