from autograd import numpy as np
from vitamine.bundle_adjustment.triangulation import (
    points_from_unknown_poses, points_from_known_poses)
from vitamine.bundle_adjustment.mask import correspondence_mask, fill_masked
from vitamine.bundle_adjustment.pnp import estimate_pose, estimate_poses
from vitamine.correspondences import count_correspondences
from vitamine.rigid.rotation import rodrigues


class InitialViewpointFinder(object):
    def __init__(self, keypoints):
        pass

    def compute(self):
        viewpoint1, viewpoint2 = 0, 1
        return viewpoint1, viewpoint2


def estimate_pose(points, keypoints_, K):
    omega, translation = estimate_pose(points, keypoints_, self.K)
    return rodrigues(omega.reshape(1, -1))[0], translation


def select_next_viewpoint(points, keypoints, used_viewpoints):
    assert(isinstance(used_viewpoints, set))

    count = count_correspondences(points, keypoints)

    # sort viewpoints in the descending order by correspondences
    viewpoints = np.argsort(count)[::-1]
    for v in viewpoints:
        # return the first element not contained in 'used_viewpoints'
        if v not in used_viewpoints:
            return v
    raise ValueError


class Initializer(object):
    def __init__(self, keypoints, K, viewpoint_finder):
        self.keypoints = keypoints
        self.viewpoint_finder = viewpoint_finder
        self.K = K

    def update(self, points, used_viewpoints):
        K = self.K
        next_viewpoint = select_next_viewpoint(points, self.keypoints,
                                               used_viewpoints)
        next_keypoints = self.keypoints[next_viewpoint]

        R0, t0 = estimate_pose(points, next_keypoints, self.K)

        # run triangulation for all used viewpoints
        for viewpoint in used_viewpoints:
            keypoints_ = self.keypoints[viewpoint]
            mask = correspondence_mask(next_keypoints, keypoints_)

            R1, t1 = estimate_pose(points, keypoints_, self.K)
            points_, depths_are_valid = points_from_known_poses(
                R0, R1, t0, t1, next_keypoints[mask], keypoints_[mask], self.K)
            points[mask] = points_

        used_viewpoints.add(next_viewpoint)

        return points, used_viewpoints

    def initialize(self, viewpoint1, viewpoint2):
        assert(viewpoint1 != viewpoint2)

        n_viewpoints = self.keypoints.shape[0]
        assert(0 <= viewpoint1 < n_viewpoints)
        assert(0 <= viewpoint2 < n_viewpoints)

        mask = correspondence_mask(
            self.keypoints[viewpoint1],
            self.keypoints[viewpoint2]
        )

        # create the initial points used to determine relative poses of
        # other viewpoints than viewpoint1 and viewpoint2
        R, t, points_ = points_from_unknown_poses(
            self.keypoints[viewpoint1, mask],
            self.keypoints[viewpoint2, mask],
            self.K
        )
        points = fill_masked(points_, mask)

        # represented in a set
        used_viewpoints = {viewpoint1, viewpoint2}

        while len(used_viewpoints) < n_viewpoints:
            points, used_viewpoints = self.update(points, used_viewpoints)
        omegas, translations = estimate_poses(points, self.keypoints, self.K)
        return omegas, translations, points


        #   points = points_from_unknown_poses(viewpoint1, viewpoint2)
        #   used_viewpoints = [viewpoint1, viewpoint2]
        #   while len(used_viewpoints) < n_viewpoints:
        #       next_viewpoint = select_next_viewpoint(points, used_viewpoints)
        #       for viewpoint in used_viewpoints:#
        #           points_ = points_from_known_poses(next_viewpoint, viewpoint)
        #           points = merge(points, points_)
        #       used_viewpoints.append(next_viewpoint)

        # run PnP to estimate relative poses of cameras from the point cloud
