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


def estimate_pose_(points, keypoints_, K):
    omega, translation = estimate_pose(points, keypoints_, K)
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


class PointUpdater(object):
    def __init__(self, K):
        self.K = K

    def update(self, points, existing_keypoints, new_keypoints):
        R0, t0 = estimate_pose_(points, new_keypoints, self.K)

        for keypoints_ in existing_keypoints:
            R1, t1 = estimate_pose_(points, keypoints_, self.K)

            mask = correspondence_mask(new_keypoints, keypoints_)

            # HACK should we check 'depths_are_valid' ?
            points[mask], depths_are_valid = points_from_known_poses(
                R0, R1, t0, t1,
                new_keypoints[mask], keypoints_[mask], self.K
            )
        return points


class Initializer(object):
    def __init__(self, keypoints, K):
        self.keypoints = keypoints
        self.K = K
        self.updater = PointUpdater(K)

    def initialize(self, viewpoint1, viewpoint2):
        assert(viewpoint1 != viewpoint2)

        n_viewpoints = self.keypoints.shape[0]
        assert(0 <= viewpoint1 < n_viewpoints)
        assert(0 <= viewpoint2 < n_viewpoints)

        mask = correspondence_mask(self.keypoints[viewpoint1],
                                   self.keypoints[viewpoint2])

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
            next_viewpoint = select_next_viewpoint(points, self.keypoints,
                                                   used_viewpoints)
            points = self.updater.update(
                points,
                self.keypoints[sorted(list(used_viewpoints))],
                self.keypoints[next_viewpoint])
            used_viewpoints.add(next_viewpoint)

        omegas, translations = estimate_poses(points, self.keypoints, self.K)
        return omegas, translations, points
