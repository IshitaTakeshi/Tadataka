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


camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])
projection = PerspectiveProjection(camera_parameters)

points_true = np.array([
   [4, -1, 3],   # 0
   [1, -3, 0],   # 1
   [-2, 0, -6],  # 2
   [0, 0, 0],    # 3
   [-3, -2, -5], # 4
   [-3, -1, 8],  # 5
   [-4, -2, 3],  # 6
   [4, 0, 1],    # 7
   [-2, 1, 1],   # 8
   [4, 1, 6],    # 9
   [-4, 4, -1],  # 10
   [-5, 3, 3],   # 11
   [-1, 3, 2],   # 12
   [2, -3, -5]   # 13
])

omegas = np.array([
    [0, 0, 0],
    [0, np.pi / 2, 0],
    [np.pi / 2, 0, 0],
    [0, np.pi / 4, 0],
    [0, -np.pi / 4, 0],
    [-np.pi / 4, np.pi / 4, 0],
    [0, np.pi / 8, -np.pi / 4]
])


rotations = rodrigues(omegas)
translations = generate_translations(rotations, points_true)
keypoints_true, positive_depth_mask = generate_observations(
    rotations, translations, points_true, projection
)

# generate dummy descriptors
# allocate sufficient lengths of descriptors for redundancy
descriptors = random_binary((len(points_true), 256))


def test_find_best_match():
    descriptors_ = [
        descriptors[0:3],  # 3 points can match
        add_noise(descriptors, [1, 3, 6]),  # 11 points can match
        descriptors[4:8],  # 4 points can match
        add_noise(descriptors, np.arange(6, 11))  # 9 points can match
    ]

    expected = np.vstack((
        [0, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13],
        [0, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13]
    )).T
    matches01, argmax = find_best_match(match, descriptors_, descriptors)
    assert_array_equal(matches01, expected)
    assert(argmax == 1)


def test_try_initialize_from_two():
    vo = VisualOdometry(camera_parameters, FOV(0.0))
    vo.init_first_keyframe(observations[0], descriptors)
    assert_array_equal(vo.poses[0].R, np.identity(3))
    assert_array_equal(vo.poses[0].t, np.zeros(3))
    vo.try_initialize_from_two(observations[1], descriptors)


def test_triangulation():
    pass
