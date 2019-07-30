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


class Initializer(object):
    def __init__(self, keypoints, K):
        self.keypoints = keypoints
        self.finder = InitialViewpointFinder(keypoints)
        self.K = K

    def select_next_view(self, all_points, used_viewpoints):
        count = count_correspondences(all_points, self.keypoints)

        # sort viewpoints in the descending order by correspondences
        viewpoints = np.argsort(count)[::-1]

        # remove used viewpoints from the candidates
        return np.setdiff1d(viewpoints, used_viewpoints)[0]

    def estimate_pose(self, points, keypoints_):
        omega, translation = estimate_pose(points, keypoints_, self.K)
        return rodrigues(omega.reshape(1, -1))[0], translation

    def update(self, all_points, used_viewpoints):
        K = self.K
        next_view = self.select_next_view(all_points, used_viewpoints)
        next_keypoints = self.keypoints[next_view]

        R0, t0 = self.estimate_pose(all_points, next_keypoints)

        # run triangulation for all used viewpoints
        for viewpoint in used_viewpoints:
            keypoints_ = self.keypoints[viewpoint]
            mask = correspondence_mask(next_keypoints, keypoints_)

            R1, t1 = self.estimate_pose(all_points, keypoints_)
            points, depths_are_valid = points_from_known_poses(
                R0, R1, t0, t1,
                next_keypoints[mask], keypoints_[mask], self.K
            )
            all_points[mask] = points

        used_viewpoints.append(next_view)

        return all_points, used_viewpoints

    def initialize(self):
        viewpoint1, viewpoint2 = self.finder.compute()

        mask = correspondence_mask(
            self.keypoints[viewpoint1],
            self.keypoints[viewpoint2]
        )

        R, t, points = points_from_unknown_poses(
            self.keypoints[viewpoint1, mask],
            self.keypoints[viewpoint2, mask],
            self.K
        )
        all_points = fill_masked(points, mask)

        n_viewpoints = self.keypoints.shape[0]
        used_viewpoints = [viewpoint1, viewpoint2]

        while len(used_viewpoints) < n_viewpoints:
            all_points, used_viewpoints = self.update(
                all_points, used_viewpoints)
        omegas, translations = estimate_poses(all_points,
                                              self.keypoints, self.K)
        return omegas, translations, all_points

        #   all_points = points_from_unknown_poses(viewpoint1, viewpoint2)
        #   used_viewpoints = [viewpoint1, viewpoint2]
        #   while len(used_viewpoints) < n_viewpoints:
        #       next_view = select_next_view(all_points, used_viewpoints)
        #       for viewpoint in used_viewpoints:#
        #           points = points_from_known_poses(next_view, viewpoint)
        #           all_points = merge(all_points, points)
        #       used_viewpoints.append(next_view)

        # run PnP to estimate relative poses of cameras from the point cloud
