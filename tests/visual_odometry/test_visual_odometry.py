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
from vitamine.visual_odometry.visual_odometry import (
    Triangulation, VisualOdometry, initialize)
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
    [0, np.pi / 2, 0]
])

rotations = rodrigues(omegas)
translations = generate_translations(rotations, points)
observations, positive_depth_mask = generate_observations(
    rotations, translations, points, projection
)

# generate dummy descriptors
# allocate sufficient lengths of descriptors for redundancy
descriptors = random_binary((len(points), 256))


def test_triangulation():
    R1, R2 = rotations
    t1, t2 = translations
    triangulation = Triangulation(R1, R2, t1, t2)

    descriptors1 = np.copy(descriptors)
    descriptors2 = np.copy(descriptors)
    keypoints1, keypoints2 = observations[0:2]

    matches, points = triangulation.triangulate(keypoints1, keypoints2,
                                                descriptors1, descriptors2)
    P = transform_all(np.array([R1, R2]), np.array([t1, t2]), points)
    assert_array_almost_equal(projection.compute(P[0]), keypoints1)
    assert_array_almost_equal(projection.compute(P[1]), keypoints2)
