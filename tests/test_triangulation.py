import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_equal, assert_almost_equal)

from projection.projections import PerspectiveProjection
from camera import CameraParameters
from rigid.rotation import tangent_so3
from rigid.transformation import transform_each
from triangulation import (estimate_fundamental, extract_poses,
                           projection_matrix, structure_from_poses)
from matrix import to_homogeneous


X_true = np.array([
   [2, -2, 2],
   [1, -3, -2],
   [-2, 3, -2],
   [-3, -2, -5],
   [-3, -1, 2],
   [-4, -2, 3],
   [4, 1, 1],
   [-2, 3, 1],
   [4, 1, 2],
   [-4, 4, -1]
])

rotations = np.array([
    [[1, 0, 0],
     [0, 0, -1],
     [0, 1, 0]]
])
translations = np.array([[-3, 1, 4]])

# skew matrices corresponding to each translation
skews = tangent_so3(translations)

camera_parameters = CameraParameters(focal_length=[0.8, 1.2], offset=[0.8, 0.2])
projection = PerspectiveProjection(camera_parameters)


def normalize(M):
    m = M.flatten()
    norm = np.linalg.norm(m)
    return M / (norm * np.sign(m[-1]))


def test_estimate_fundamental():
    X = transform_each(rotations, translations, X_true)[0]

    points0 = projection.project(X_true)
    points1 = projection.project(X)

    F = estimate_fundamental(points0, points1)

    N = X_true.shape[0]
    for i in range(N):
        x0 = np.append(points0[i], 1)
        x1 = np.append(points1[i], 1)
        assert_almost_equal(x1.dot(F).dot(x0), 0)


def test_structure_from_poses():
    X = transform_each(rotations, translations, X_true)[0]

    points0 = projection.project(X_true)
    points1 = projection.project(X)

    R0 = np.identity(3)
    t0 = np.zeros(3)

    R1 = rotations[0]
    t1 = translations[0]

    K = camera_parameters.matrix

    N = X_true.shape[0]
    for i in range(N):
        x = structure_from_poses(K, R0, R1, t0, t1, points0[i], points1[i])
        assert_array_almost_equal(x[:3], X_true[i])
        assert_equal(x[3], 1)


def test_extract_poses():
    def to_essential(R, t):
        S = tangent_so3(t.reshape(1, 3))[0]
        return np.dot(S, R)

    E_true = np.dot(skews[0], rotations[0])
    R1, R2, t1, t2 = extract_poses(E_true)

    # make sure that both of R1 and R2 are rotation matrices
    assert_array_almost_equal(np.dot(R1.T, R1), np.identity(3))
    assert_array_almost_equal(np.dot(R2.T, R2), np.identity(3))
    assert_almost_equal(np.linalg.det(R1), 1.)
    assert_almost_equal(np.linalg.det(R2), 1.)

    E_pred = to_essential(R1, t1)
    assert_array_almost_equal(E_pred / np.linalg.norm(E_pred),
                              E_true / np.linalg.norm(E_true))

    E_pred = to_essential(R2, t2)
    assert_array_almost_equal(E_pred / np.linalg.norm(E_pred),
                              E_true / np.linalg.norm(E_true))


def test_projection_matrix():
    pass
