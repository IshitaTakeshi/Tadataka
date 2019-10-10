from numpy.testing import assert_array_almost_equal, assert_array_equal
from autograd import numpy as np

from vitamine.visual_odometry.local_ba import LocalBundleAdjustment, Projection
from tests.utils import unit_uniform


def to_poses(omegas, translations):
    return np.hstack((omegas, translations))


def test_jacobian():
    n_viewpoints = 3
    n_points = 4

    points = 10 * unit_uniform((n_points, 3))
    omegas = np.pi * unit_uniform((n_viewpoints, 3))
    translations = 10 * unit_uniform((n_viewpoints, 3))
    poses = to_poses(omegas, translations)

    dpoints = 0.01 * unit_uniform(points.shape)
    domegas = 0.01 * np.pi * unit_uniform(omegas.shape)
    dtranslations = 0.01 * unit_uniform(translations.shape)
    dposes = to_poses(domegas, dtranslations)

    # assume that all points are visible
    mask = np.ones((n_viewpoints, n_points))
    viewpoint_indices, point_indices = np.where(mask)

    projection = Projection(viewpoint_indices, point_indices)
    A, B = projection.jacobians(poses, points)

    Q = projection.compute

    # test sign(Q(a + da, b) - Q(a, b)) == sign(A * da)
    # where A = dQ / da
    dx_true = Q(poses + dposes, points) - Q(poses, points)
    for index, j in enumerate(viewpoint_indices):
        dx_pred = A[index].dot(dposes[j])
        assert_array_equal(np.sign(dx_true[index]), np.sign(dx_pred))

    # test sign(Q(a, b + db) - Q(a, b)) == sign(B * db)
    # where B = dQ / db
    dx_true = Q(poses, points + dpoints) - Q(poses, points)
    for index, i in enumerate(point_indices):
        dx_pred = B[index].dot(dpoints[i])
        assert_array_equal(np.sign(dx_true[index]), np.sign(dx_pred))


def add_noise(array, scale):
    return array + scale * unit_uniform(array.shape)


def test_local_bundle_adjustment():
    def error(keypoints1, keypoints2):
        return np.power(keypoints1 - keypoints2, 2).sum()

    def run(omegas1, translations1, points1):
        keypoints1 = projection.compute(
            to_poses(omegas1, translations1), points1
        )

        # refine parameters
        omegas2, translations2, points2 = local_ba.compute(
            omegas1, translations1, points1)

        keypoints2 = projection.compute(
            to_poses(omegas2, translations2), points2
        )

        # error shoud be decreased after bundle adjustment
        E1 = error(keypoints1, keypoints_true)
        E2 = error(keypoints2, keypoints_true)

        if np.isclose(E1, E2):
            # nothing updated
            assert_array_equal(omegas1, omegas2)
            assert_array_equal(translations1, translations2)
            assert_array_equal(points1, points2)
            return

        assert(E2 < E1)


    n_viewpoints = 4
    n_points = 5

    mask = np.ones((n_viewpoints, n_points))

    viewpoint_indices, point_indices = np.where(mask)

    projection = Projection(viewpoint_indices, point_indices)

    omegas_true = np.pi * unit_uniform((n_viewpoints, 3))
    translations_true = unit_uniform((n_viewpoints, 3))
    points_true = unit_uniform((n_points, 3))

    keypoints_true = projection.compute(
        to_poses(omegas_true, translations_true), points_true
    )

    local_ba = LocalBundleAdjustment(viewpoint_indices, point_indices,
                                     keypoints_true)

    omegas_noisy = add_noise(omegas_true, 0.01 * np.pi)
    translations_noisy = add_noise(translations_true, 0.01)
    points_noisy = add_noise(points_true, 0.01)

    # check BA can refine point / pose parameters for each case
    # if only omegas are noisy
    run(omegas_noisy, translations_true, points_true)
    # if only translations are noisy
    run(omegas_true, translations_noisy, points_true)
    # if only points are noisy
    run(omegas_true, translations_true, points_noisy)
    # if all parameters are noisy
    run(omegas_noisy, translations_noisy, points_noisy)
