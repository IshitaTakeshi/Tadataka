from autograd import numpy as np

from vitamine.exceptions import InvalidDepthsException, print_error
from vitamine import triangulation as TR
from vitamine.pose import Pose


def depth_mask_condition(mask, min_positive_dpth_ratio=0.8):
    return np.sum(mask) / len(mask) >= min_positive_dpth_ratio


def pose_point_from_keypoints(keypoints0, keypoints1, matches01):
    R, t, points, valid_depth_mask = TR.pose_point_from_keypoints(
        keypoints0[matches01[:, 0]],
        keypoints1[matches01[:, 1]]
    )

    if not depth_mask_condition(valid_depth_mask):
        raise InvalidDepthsException(
            "Most of points are behind cameras. Maybe wrong matches?"
        )

    return Pose(R, t), points[valid_depth_mask], matches01[valid_depth_mask]


def points_from_known_poses(keypoints0, keypoints1, pose0, pose1, matches01):
    points, valid_depth_mask = TR.points_from_known_poses(
        pose0.R, pose1.R, pose0.t, pose1.t,
        keypoints0[matches01[:, 0]],
        keypoints1[matches01[:, 1]]
    )

    if not depth_mask_condition(valid_depth_mask):
        raise InvalidDepthsException(
            "Most of points are behind cameras. Maybe wrong matches?"
        )

    return points[valid_depth_mask], matches01[valid_depth_mask]


def triangulation(matcher, points,
                  pose_list, local_features_list, pose0, lf0):
    assert(len(local_features_list) == len(pose_list))
    # any keypoints don't have corresponding 3D points
    assert(np.all(~lf0.is_triangulated))

    # we focus on only untriangulated points
    kd0 = lf0.get()

    P = []
    # triangulate untriangulated points
    for i, (lf1, pose1) in enumerate(zip(local_features_list, pose_list)):
        kd1 = lf1.untriangulated()
        if len(kd1.keypoints) == 0:
            continue

        matches01 = matcher(kd0, kd1)

        P.append((lf1, matches01))

        # keep elements that are not triangulated yet
        mask = ~lf0.is_triangulated[matches01[:, 0]]
        if np.sum(mask) == 0:
            # all matched keypoints are already triangulated
            continue

        matches01 = matches01[mask]

        try:
            points_, matches01 = points_from_known_poses(
                kd0.keypoints, kd1.keypoints,
                pose0, pose1, matches01
            )
        except InvalidDepthsException as e:
            print_error(str(e))
            raise InvalidDepthsException(
                f"Failed to triangulate with {i}th pose"
            )

        point_indices_ = points.add(points_)
        lf0.point_indices[matches01[:, 0]] = point_indices_

    # copy point indices back to each lf1
    for lf1, matches01 in P:
        indices0, indices1 = matches01[:, 0], matches01[:, 1]
        lf1.associate_points(indices1, lf0.point_indices[indices0])


def copy_triangulated(matcher, local_features_list, lf1):
    for lf0 in local_features_list:
        # copy point indices from lf0 to lf1
        matches01 = matcher(lf0.triangulated(), lf1.untriangulated())
        if len(matches01) == 0:
            continue
        point_indices = lf0.triangulated_point_indices(matches01[:, 0])
        lf1.associate_points(matches01[:, 1], point_indices)
