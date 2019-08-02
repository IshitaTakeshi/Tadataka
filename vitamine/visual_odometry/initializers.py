from autograd import numpy as np
from vitamine.bundle_adjustment.triangulation import (
    points_from_unknown_poses, points_from_known_poses, MultipleTriangulation)
from vitamine.bundle_adjustment.mask import (
    correspondence_mask, fill_masked, point_mask)
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


def select_new_viewpoint(points, keypoints, used_viewpoints):
    assert(isinstance(used_viewpoints, set))

    count = count_correspondences(points, keypoints)

    # sort viewpoints in the descending order by correspondences
    viewpoints = np.argsort(count)[::-1]
    for v in viewpoints:
        # return the first element not contained in 'used_viewpoints'
        if v not in used_viewpoints:
            return v
    raise ValueError


def create_empty(shape):
    return np.full(shape, np.nan)


def update(all_points, points):
    mask = point_mask(points)
    all_points[mask] = points[mask]
    return all_points


class Initializer(object):
    def __init__(self, keypoints, K):
        self.keypoints = keypoints
        self.K = K

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

        all_points = create_empty((self.keypoints.shape[1], 3))
        all_points = update(all_points, points)

        omegas, translations = None, None
        while len(used_viewpoints) < n_viewpoints:
            used_keypoints = self.keypoints[sorted(used_viewpoints)]

            omegas, translations = estimate_poses(
                all_points, used_keypoints, self.K)
            triangulation = MultipleTriangulation(
                rodrigues(omegas), translations,
                used_keypoints,
                self.K
            )

            new_viewpoint = select_new_viewpoint(all_points, self.keypoints,
                                                 used_viewpoints)
            assert(new_viewpoint not in used_viewpoints)
            new_keypoints = self.keypoints[new_viewpoint]

            omega, translation = estimate_pose(all_points, new_keypoints, self.K)
            points = triangulation.triangulate(
                rodrigues(omega.reshape(1, -1))[0],
                translation,
                new_keypoints
            )

            used_viewpoints.add(new_viewpoint)

            all_points = update(all_points, points)

        omegas, translations = estimate_poses(points, self.keypoints, self.K)
        return omegas, translations, all_points
