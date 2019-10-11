from autograd import numpy as np

from vitamine.exceptions import InvalidDepthsException, print_error
from vitamine import triangulation as TR
from vitamine.pose import Pose
from vitamine.keypoints import filter_matches


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
        np.set_printoptions(suppress=True)
        raise InvalidDepthsException(
            (f"Most of points are behind cameras. Maybe wrong matches? "
             f"valid_depth_mask = {valid_depth_mask}")
        )

    return points[valid_depth_mask], matches01[valid_depth_mask]


def triangulation(points, matches,
                  pose_list, keypoints_list, point_indices_list,
                  pose0, keypoints0, point_indices0):
    assert(len(point_indices_list) == len(pose_list))
    # any keypoints don't have corresponding 3D points

    P = []
    Z = zip(matches, pose_list, keypoints_list, point_indices_list)
    for matches01, pose1, keypoints1, point_indices1 in Z:
        if len(matches01) == 0:
            continue

        P.append((point_indices1, matches01))

        # use keypoints that are not triangulated yet
        matches01 = filter_matches(matches01,
                                   ~point_indices0.is_triangulated,
                                   ~point_indices1.is_triangulated)

        if len(matches01) == 0:
            continue

        try:
            points_, matches01 = points_from_known_poses(
                keypoints0, keypoints1,
                pose0, pose1, matches01
            )
        except InvalidDepthsException as e:
            print_error(str(e))
            raise InvalidDepthsException("Failed to triangulate")

        point_indices = points.add(points_)
        point_indices0.subscribe(matches01[:, 0], point_indices)


    for point_indices1, matches01 in P:
        matches01 = filter_matches(matches01,
                                   point_indices0.is_triangulated,
                                   ~point_indices1.is_triangulated)

        indices0, indices1 = matches01[:, 0], matches01[:, 1]
        point_indices1.subscribe(indices1, point_indices0.get(indices0))


def copy_triangulated(matches, point_indices_list, point_indices0):
    # matched keypoints indicate the same 3D point
    # so we copy 3D point indices in existing frames to the newly added frame
    # so that the new frame can indicate the existing 3D points
    for point_indices1, matches01 in zip(point_indices_list, matches):
        if len(matches01) == 0:
            continue
        # copy from triangulated point_indices1
        # to untriangulated point_indices0
        matches01 = filter_matches(matches01,
                                   ~point_indices0.is_triangulated,
                                   point_indices1.is_triangulated)
        if len(matches01) == 0:
            continue

        indices0, indices1 = matches01[:, 0], matches01[:, 1]
        point_indices0.subscribe(indices0, point_indices1.get(indices1))

    return point_indices0
