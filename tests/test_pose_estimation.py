import pytest

from autograd import numpy as np
from numpy.testing import assert_array_almost_equal

from vitamine.dataset.observations import generate_translations
from vitamine.so3 import rodrigues

from vitamine.exceptions import NotEnoughInliersException
from vitamine.camera import CameraParameters
from vitamine.projection import PerspectiveProjection
from vitamine.dataset.observations import generate_observations
from vitamine.pose_estimation import solve_pnp


camera_parameters = CameraParameters(focal_length=[1, 1], offset=[0, 0])
projection = PerspectiveProjection(camera_parameters)

points = np.array([
   [4, -1, 3],
   [1, -3, -2],
   [-2, 3, -2],
   [-3, -2, -5],
   [4, 1, 1],
   [-2, 3, 1],
   [-4, 4, -1]
])

omegas = np.pi * np.array([
    [0, 0, 0],
    [0, 1 / 2, 0],
    [0, 1 / 4, 1 / 4],
    [1 / 8, -1 / 4, 0],
])

# translations = generate_translations(rodrigues(omegas), points)
translations = np.array([
    [ 3.0, 2.0, 7.0],
    [-1.0, -1.0, 6.0],
    [-1.5, 2.0, 5.0],
    [-2.0, -0.5, 8.0]
])


keypoints, positive_depth_mask = generate_observations(
    rodrigues(omegas), translations, points, projection
)

def test_solve_pnp():
    for keypoints_, omega_true, t_true in zip(keypoints, omegas, translations):
        omega_pred, t_pred = solve_pnp(points, keypoints_)
        assert_array_almost_equal(omega_true, omega_pred)
        assert_array_almost_equal(t_true, t_pred)

    keypoints0 = keypoints[0]
    # 4 correspondences
    # this should be able to run
    solve_pnp(points[0:4], keypoints0[0:4])

    with pytest.raises(NotEnoughInliersException):
        # not enough correspondences
        solve_pnp(points[0:3], keypoints0[0:3])
