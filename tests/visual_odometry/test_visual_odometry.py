from autograd import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from vitamine.so3 import rodrigues
from vitamine.projection import PerspectiveProjection
from vitamine.dataset.points import cubic_lattice
from vitamine.dataset.observations import (
    generate_observations, generate_translations)
from vitamine.visual_odometry import visual_odometry
from vitamine.camera import CameraParameters
from vitamine.camera_distortion import FOV
from vitamine.keypoints import  match
from vitamine.visual_odometry.visual_odometry import (
    Triangulation, VisualOdometry)
from vitamine.rigid.transformation import transform_all


def random_binary(size):
    return np.random.randint(0, 2, size, dtype=np.bool)


def add_noise(descriptors, indices):
    descriptors = np.copy(descriptors)
    descriptors[indices] = random_binary((len(indices), descriptors.shape[1]))
    return descriptors


camera_parameters = CameraParameters(
    focal_length=[1., 1.],
    offset=[0., 0.]
)

projection = PerspectiveProjection(camera_parameters)

points = cubic_lattice(3)

omegas = np.array([
    [0, 0, 0],
    [0, np.pi / 2, 0],
    [np.pi / 2, 0, 0],
    [0, np.pi / 4, 0],
    [0, -np.pi / 4, 0],
    [-np.pi / 4, 0, 0],
    [np.pi / 2, 0, 0]
])


rotations = rodrigues(omegas)
translations = generate_translations(rotations, points)
observations, positive_depth_mask = generate_observations(
    rotations, translations, points, projection
)

# generate dummy descriptors
# allocate sufficient lengths of descriptors for redundancy
descriptors = random_binary((len(points), 256))


def test_init_points():
    keypoints0, keypoints1 = observations[0:2]

    vo = VisualOdometry(camera_parameters, FOV(0.0))
    vo.try_add(keypoints0, np.copy(descriptors))
    vo.try_add(keypoints1, np.copy(descriptors))

    rotations, translations = vo.poses
    P = transform_all(rotations, translations, vo.points)
    assert_array_almost_equal(projection.compute(P[0]), keypoints0)
    assert_array_almost_equal(projection.compute(P[1]), keypoints1)

    # randomize first 10 descriptors
    vo = VisualOdometry(camera_parameters, FOV(0.0))
    vo.try_add(keypoints0, add_noise(descriptors, np.arange(0, 10)))
    vo.try_add(keypoints1, add_noise(descriptors, np.arange(0, 10)))

    rotations, translations = vo.poses
    P = transform_all(rotations, translations, vo.points)
    assert_array_almost_equal(projection.compute(P[0]), keypoints0[10:])
    assert_array_almost_equal(projection.compute(P[1]), keypoints1[10:])


def test_try_add():
    # test the case all descriptors are same
    vo = VisualOdometry(camera_parameters, FOV(0.0))
    for i in range(3):
        vo.try_add(observations[i], np.copy(descriptors))

    rotations, translations = vo.poses
    P = transform_all(rotations, translations, vo.points)
    for i in range(3):
        assert_array_almost_equal(projection.compute(P[i]), observations[i])

    vo = VisualOdometry(camera_parameters, FOV(0.0))

    descriptors0 = descriptors  # nothing modified
    descriptors1 = add_noise(descriptors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    descriptors2 = add_noise(descriptors, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
    descriptors3 = add_noise(descriptors, [10, 11, 12, 13, 14])

    # descriptors0[10:27] and descriptors1[10:27] should be matched and
    # triangulated
    vo.try_add(observations[0], descriptors0)
    vo.try_add(observations[1], descriptors1)
    # descriptors0[0:10] and descriptors1[0:10] cannot be matched
    assert_array_equal(vo.get_untriangulated(0), np.arange(10))
    # descriptors0[5:10] and descriptors2[5:10] should be matched and the
    # corresponding keypoints should be triangulated
    # descriptors0[15:27] and descriptors2[15:27] have the same
    # descriptors but the corresponding keypoints are already
    # triangulated so they are ignored
    vo.try_add(observations[2], descriptors2)
    # descriptors0[0:5] and descriptors3[0:5] should be matched
    vo.try_add(observations[3], descriptors3)

    rotations, translations = vo.poses
    P = transform_all(rotations, translations, vo.points)
    # points corresponding to descriptors0[10:27] and descriptors1[10:27]
    assert_array_almost_equal(projection.compute(P[0, 0:17]),
                              observations[0, 10:])
    assert_array_almost_equal(projection.compute(P[1, 0:17]),
                              observations[1, 10:])
    # points corresponding to descriptors2[5:10] and descriptors0[5:10]
    assert_array_almost_equal(projection.compute(P[0, 17:22]),
                              observations[0, 5:10])
    assert_array_almost_equal(projection.compute(P[2, 17:22]),
                              observations[2, 5:10])
    # points corresponding to descriptors3[0:5] and descriptors0[0:5]
    assert_array_almost_equal(projection.compute(P[0, 22:27]),
                              observations[0, 0:5])
    assert_array_almost_equal(projection.compute(P[3, 22:27]),
                              observations[3, 0:5])

def test_try_remove():
    vo = VisualOdometry(camera_parameters, FOV(0.0), min_active_keyframes=4)
    vo.try_add(observations[0], np.copy(descriptors))
    vo.try_add(observations[1], add_noise(descriptors, np.arange(0, 18)))
    vo.try_add(observations[2], add_noise(descriptors, np.arange(0, 15)))
    vo.try_add(observations[3], add_noise(descriptors, np.arange(0, 10)))
    assert(not vo.try_remove())  # 'vo' have to hold at least 4 keyframes
    assert(vo.reference_keyframe_id == 0)

    vo.try_add(observations[4], add_noise(descriptors, np.arange(0, 5)))
    assert(vo.try_remove())
    assert(vo.reference_keyframe_id == 1)

    vo.try_add(observations[5], add_noise(descriptors, np.arange(15, 27)))
    assert(vo.try_remove())
    assert(vo.reference_keyframe_id == 2)
