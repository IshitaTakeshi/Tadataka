from autograd import numpy as np
from numpy.testing import assert_array_almost_equal
from vitamine.so3 import rodrigues
from vitamine.projection import PerspectiveProjection
from vitamine.dataset.points import cubic_lattice
from vitamine.dataset.observations import (
    generate_observations, generate_translations)
from vitamine.visual_odometry import visual_odometry
from vitamine.camera import CameraParameters
from vitamine.visual_odometry.visual_odometry import Triangulation
from vitamine.rigid.transformation import transform_all

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
descriptors = np.random.randint(0, 2, (len(points), 128))


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
