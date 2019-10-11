from autograd import numpy as np
from vitamine.pose import Pose, solve_pnp
from vitamine.exceptions import NotEnoughInliersException


def get_correspondences(matches, point_indices_list):
    assert(len(point_indices_list) == len(matches))
    assert(len(point_indices_list) > 0)

    point_indices = []
    keypoint_indices = []
    for point_indices1, matches01 in zip(point_indices_list, matches):
        if len(matches01) == 0:
            continue

        mask = point_indices1.is_triangulated[matches01[:, 1]]
        if np.sum(mask) == 0:
            continue

        indices0, indices1 = matches01[mask, 0], matches01[mask, 1]
        keypoint_indices.append(indices0)
        point_indices.append(point_indices1.get(indices1))
    return np.concatenate(point_indices), np.concatenate(keypoint_indices)


def estimate_pose(points, matches, point_indices_list, keypoints0):
    point_indices, keypoint_indices = get_correspondences(
        matches, point_indices_list
    )
    points_ = points.get(point_indices)
    keypoints_ = keypoints0[keypoint_indices]

    try:
        return solve_pnp(points_, keypoints_)
    except NotEnoughInliersException:
        return None
