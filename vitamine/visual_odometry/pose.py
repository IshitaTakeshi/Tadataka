from autograd import numpy as np
from vitamine import pose_estimation as PE
from vitamine.pose import Pose
from vitamine.visual_odometry.keypoint import is_triangulated
from vitamine.exceptions import NotEnoughInliersException


def get_correspondences(matches, point_indices_list):
    assert(len(point_indices_list) == len(matches))
    assert(len(point_indices_list) > 0)

    point_indices = []
    keypoint_indices = []
    for point_indices1, matches01 in zip(point_indices_list, matches):
        if len(matches01) == 0:
            continue

        mask = is_triangulated(point_indices1)[matches01[:, 1]]
        if np.sum(mask) == 0:
            continue

        indices0, indices1 = matches01[mask, 0], matches01[mask, 1]
        keypoint_indices.append(indices0)
        point_indices.append(point_indices1[indices1])
    return np.concatenate(point_indices), np.concatenate(keypoint_indices)


def estimate_pose(points, matches, point_indices_list, keypoints0):
    point_indices, keypoint_indices = get_correspondences(
        matches, point_indices_list
    )
    points_ = points.get(point_indices)
    keypoints_ = keypoints0[keypoint_indices]

    try:
        omega, t = PE.solve_pnp(points_, keypoints_)
    except NotEnoughInliersException:
        return None

    return Pose(omega, t)
