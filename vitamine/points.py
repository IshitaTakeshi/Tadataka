from collections import defaultdict
import warnings

from autograd import numpy as np

from vitamine.exceptions import InvalidDepthException, print_error
from vitamine.triangulation import (linear_triangulation,
                                    pose_point_from_keypoints)


def init_empty():
    return np.empty((0, 3), dtype=np.float64)


def warn_if_incorrect_match(index_map, viewpoint0, viewpoint1,
                            keypoint_index0, keypoint_index1):
    point_index0 = index_map[viewpoint0][keypoint_index0]
    point_index1 = index_map[viewpoint1][keypoint_index1]

    if point_index0 != point_index1:
        warnings.warn(
            f"Wrong match! "
            f"Keypoint index {keypoint_index0} in {viewpoint0} and "
            f"keypoint index {keypoint_index1} in {viewpoint1} have matched, "
            f"but indicate different 3D points!"
        )


class Triangulation(object):
    def __init__(self, pose0, pose1, keypoints0, keypoints1):
        self.pose0, self.pose1 = pose0, pose1
        self.keypoints0, self.keypoints1 = keypoints0, keypoints1

    def triangulate(self, index0, index1):
        keypoint0 = self.keypoints0[index0]
        keypoint1 = self.keypoints1[index1]

        try:
            return linear_triangulation(self.pose0, self.pose1,
                                        keypoint0, keypoint1)
        except InvalidDepthException as e:
            print_error(e)
            return None


class PointManager(object):
    def __init__(self):
        # {viewpoint index: {keypoint index: point index}}
        self.index_map = defaultdict(dict)
        self.points = dict()

    def export_points(self):
        return np.array(list(self.points.values()))

    def overwrite(self, point_index, point):
        self.points[point_index] = point

    def new_point_index(self):
        point_indices = self.points.keys()
        if len(point_indices) == 0:
            return 0
        return max(point_indices) + 1

    def add_point(self, point, viewpoint0, viewpoint1,
                  keypoint_index0, keypoint_index1):
        point_index = self.new_point_index()
        self.points[point_index] = point
        self.index_map[viewpoint0][keypoint_index0] = point_index
        self.index_map[viewpoint1][keypoint_index1] = point_index

    def keyerror_if_viewpoint_not_exist(self, viewpoint):
        # we need this function because index_map is 'defaultdict'
        if viewpoint not in self.index_map.keys():
            raise KeyError(f"viewpoint {viewpoint}")

    def get(self, viewpoint, keypoint_index):
        self.keyerror_if_viewpoint_not_exist(viewpoint)
        point_index = self.index_map[viewpoint][keypoint_index]
        return self.points[point_index]

    def initialize(self, keypoints0, keypoints1, matches01,
                   viewpoint0, viewpoint1):
        # no viewpoints added so far
        assert(len(self.index_map.keys()) == 0)

        pose0, pose1, points, valid_matches01 = pose_point_from_keypoints(
            keypoints0, keypoints1, matches01
        )

        for point, (index0, index1) in zip(points, valid_matches01):
            point_index = self.add_point(point, viewpoint0, viewpoint1,
                                         index0, index1)

        return pose0, pose1

    def associate_existing(self, src_viewpoint, dst_viewpoint,
                           src_keypoint_index, dst_keypoint_index):
        point_index = self.index_map[src_viewpoint][src_keypoint_index]
        # we prevent the case that one view has two keypoints that is
        # corresponding to one 3D point
        # e.g. index_map[1][4] == index_map[1][3] == 2 should not happen
        # because this means both of 4th keypoint and 3rd keypoint are
        # the projections of the 2nd 3D point in the 1st view
        if point_index in self.index_map[dst_viewpoint].values():
            warnings.warn(
                f"point index {point_index} already exists "
                f"in viwepoint {dst_viewpoint}"
            )
            return
        self.index_map[dst_viewpoint][dst_keypoint_index] = point_index

    def both_observed(self, triangulation,
                      viewpoint0, viewpoint1, indices0, indices1):
        # Triangulate in the assumption that
        # both viewpoints have already subscribed
        # There's a constraint that one 3D point have
        # only one corresponding keypoint in a frame

        keypoint_indices0 = self.index_map[viewpoint0].keys()
        keypoint_indices1 = self.index_map[viewpoint1].keys()

        for index0, index1 in zip(indices0, indices1):
            is_trinagulated0 = index0 in keypoint_indices0
            is_trinagulated1 = index1 in keypoint_indices1

            if is_trinagulated0 and is_trinagulated1:
                # Both matched keypoints have already triangulated.
                # In this case, index0 and index1 should indicate
                # the same 3D point otherwise it is a wrong match
                warn_if_incorrect_match(self.index_map, viewpoint0, viewpoint1,
                                        index0, index1)
                continue

            if is_trinagulated0:  # is_trinagulated1 == False
                # keypoint corresponding to index0 is already triangulated
                # index1 is indicating a keypoint in viewpoint1
                # that is extracted but have not been matched
                self.associate_existing(viewpoint0, viewpoint1, index0, index1)
                continue

            if is_trinagulated1:  # is_trinagulated0 == False
                # keypoint corresponding to index1 is already triangulated
                # index0 is indicating a keypoint in viewpoint1
                # that is extracted but have not been matched
                self.associate_existing(viewpoint1, viewpoint0, index1, index0)
                continue

            # neither index0 nor index1 has been triangulated
            point = triangulation.triangulate(index0, index1)
            if point is None:
                continue

            self.add_point(point, viewpoint0, viewpoint1, index0, index1)

    def either_one_observed(self, triangulation,
                            src_viewpoint, dst_viewpoint, src_indices, dst_indices):
        # assume src_viewpoint have already been observed

        src_keypoint_indices = self.index_map[src_viewpoint].keys()
        for src_index, dst_index in zip(src_indices, dst_indices):
            if src_index in src_keypoint_indices:
                # keypoint corresponding to src_index has
                # already been triangulated
                self.associate_existing(src_viewpoint, dst_viewpoint,
                                        src_index, dst_index)
                continue

            # both keypoints corresponding to src_index and dst_index
            # have not been observed
            # create it by triangulation

            point = triangulation.triangulate(src_index, dst_index)
            if point is None:
                continue

            self.add_point(point, src_viewpoint, dst_viewpoint,
                           src_index, dst_index)

    def triangulate(self, pose0, pose1,
                    keypoints0, keypoints1, matches01,
                    viewpoint0, viewpoint1):
        triangulation = Triangulation(pose0, pose1, keypoints0, keypoints1)

        viewpoints = self.index_map.keys()
        has_observed0 = viewpoint0 in viewpoints
        has_observed1 = viewpoint1 in viewpoints

        indices0, indices1 = matches01[:, 0], matches01[:, 1]

        if has_observed0 and has_observed1:
            self.both_observed(triangulation, viewpoint0, viewpoint1,
                               indices0, indices1)
            return

        if has_observed0:
            self.either_one_observed(triangulation, viewpoint0, viewpoint1,
                                     indices0, indices1)
            return

        if has_observed1:
            self.either_one_observed(triangulation, viewpoint1, viewpoint0,
                                     indices1, indices0)
            return

        raise ValueError("Neither viewpoint0 nor viewpoint1 has been observed")
