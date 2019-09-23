from autograd import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal, assert_equal)
from vitamine.so3 import rodrigues
from vitamine.projection import PerspectiveProjection
from vitamine.dataset.points import cubic_lattice
from vitamine.dataset.observations import (
    generate_observations, generate_translations)
from vitamine.visual_odometry import visual_odometry
from vitamine.camera import CameraParameters
from vitamine.camera_distortion import FOV
from vitamine.keypoints import match
from vitamine.visual_odometry.visual_odometry import (
    Triangulation, VisualOdometry, find_best_match)
from vitamine.rigid.transformation import transform_all
from tests.utils import random_binary


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



def test_find_best_match():
    descriptors_ = [
        descriptors[0:12],  # 12 points can match
        add_noise(descriptors, [1, 3, 5, 8, 9]),  # 22 points can match
        descriptors[12:27],  # 15 points can match
        add_noise(descriptors, np.arange(3, 12))  # 18 points can match
    ]

    expected = np.vstack((
        [0, 2, 4, 6, 7, 10, 11, 12, 13, 14, 15,
         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        [0, 2, 4, 6, 7, 10, 11, 12, 13, 14, 15,
         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    )).T
    matches01, argmax = find_best_match(match, descriptors_, descriptors)
    assert_array_equal(matches01, expected)
    assert_equal(argmax, 1)


def test_try_initialize_from_two():
    vo = VisualOdometry(camera_parameters, FOV(0.0))
    vo.init_first_keyframe(observations[0], descriptors)
    assert_array_equal(vo.poses[0].R, np.identity(3))
    assert_array_equal(vo.poses[0].t, np.zeros(3))
    vo.try_initialize_from_two(observations[1], descriptors)


def test_triangulation():
    pass
