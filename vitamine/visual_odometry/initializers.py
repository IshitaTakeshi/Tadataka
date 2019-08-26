from autograd import numpy as np

from vitamine.assertion import check_keypoints
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


def update(all_points, points):
    mask = point_mask(points)
    all_points[mask] = points[mask]
    return all_points


def initial_points(keypoints1, keypoints2, K):
    mask = correspondence_mask(keypoints1, keypoints2)

    # create the initial points used to determine relative poses of
    # other viewpoints than viewpoint1 and viewpoint2
    R, t, points_ = points_from_unknown_poses(keypoints1[mask],
                                              keypoints2[mask],
                                              K)
    return fill_masked(points_, mask)


class Initializer(object):
    def __init__(self, keypoints, K):
        self.keypoints = keypoints
        self.K = K

    def initialize(self, viewpoint1, viewpoint2):
        assert(viewpoint1 != viewpoint2)

        n_viewpoints = self.keypoints.shape[0]
        assert(0 <= viewpoint1 < n_viewpoints)
        assert(0 <= viewpoint2 < n_viewpoints)

        points = initial_points(self.keypoints[viewpoint1],
                                self.keypoints[viewpoint2],
                                self.K)

        # represented in a set
        used_viewpoints = {viewpoint1, viewpoint2}

        while len(used_viewpoints) < n_viewpoints:
            # select a viewpoint other than used_viewpoints
            new_viewpoint = select_new_viewpoint(points, self.keypoints,
                                                 used_viewpoints)
            assert(new_viewpoint not in used_viewpoints)

            used_keypoints = self.keypoints[sorted(used_viewpoints)]

            # triangulate the new viewpoint with existing viewpoints
            omegas, translations = estimate_poses(points, used_keypoints,
                                                  self.K)
            triangulation = MultipleTriangulation(omegas, translations,
                                                  used_keypoints, self.K)

            new_keypoints = self.keypoints[new_viewpoint]

            new_omega, new_translation = estimate_pose(points, new_keypoints,
                                                       self.K)
            points_ = triangulation.triangulate(new_omega, new_translation,
                                                new_keypoints)

            points = update(points, points_)

            used_viewpoints.add(new_viewpoint)

        omegas, translations = estimate_poses(points, self.keypoints, self.K)

        check_keypoints(self.keypoints, omegas, translations, points)
        return omegas, translations, points
