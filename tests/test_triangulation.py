import itertools

import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_equal, assert_almost_equal)
from numpy.linalg import inv, norm

from projection.projections import PerspectiveProjection
from camera import CameraParameters
from rigid.rotation import tangent_so3
from rigid.transformation import transform_each, transform
from bundle_adjustment.triangulation import (
    estimate_fundamental, fundamental_to_essential, extract_poses,
    projection_matrix, linear_triangulation)
from matrix import to_homogeneous


X_true = np.array([
   [4, -1, 3],
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
     [0, 1, 0]],
    [[0, 0, 1],
     [-1, 0, 0],
     [0, -1, 0]]
])

translations = np.array([[-3, 1, 4]])

camera_parameters = CameraParameters(focal_length=[0.8, 1.2], offset=[0.8, 0.2])
projection = PerspectiveProjection(camera_parameters)


def normalize(M):
    m = M.flatten()
    return M / (norm(m) * np.sign(m[-1]))



def test_estimate_fundamental():
    R, t = rotations[0], translations[0]
    keypoints0 = projection.project(X_true)
    keypoints1 = projection.project(transform(R, t, X_true))

    K = camera_parameters.matrix
    K_inv = np.linalg.inv(K)

    F = estimate_fundamental(keypoints0, keypoints1)
    E = fundamental_to_essential(F, K)

    N = X_true.shape[0]

    for i in range(N):
        x0 = np.append(keypoints0[i], 1)
        x1 = np.append(keypoints1[i], 1)
        assert_almost_equal(x1.dot(F).dot(x0), 0)

        y0 = np.dot(K_inv, x0)
        y1 = np.dot(K_inv, x1)
        assert_almost_equal(y1.dot(E).dot(y0), 0)

        # properties of the essential matrix
        assert_almost_equal(np.linalg.det(E), 0)
        assert_array_almost_equal(
            2 * np.dot(E, np.dot(E.T, E)) - np.trace(np.dot(E, E.T)) * E,
            np.zeros((3, 3))
        )


def to_essential(R, t):
    S = tangent_so3(t.reshape(1, 3))[0]
    return np.dot(S, R)


def test_fundamental_to_essential():
    R, t = rotations[0], translations[0]
    K0 = CameraParameters(focal_length=[0.8, 1.2], offset=[0.8, -0.2]).matrix
    K1 = CameraParameters(focal_length=[0.7, 0.9], offset=[-1.0, 0.1]).matrix

    E_true = to_essential(R, t)
    F = inv(K1).T.dot(E_true).dot(inv(K0))
    E_pred = fundamental_to_essential(F, K0, K1)
    assert_array_almost_equal(E_true, E_pred)


def test_linear_triangulation():
    R, t = rotations[0], translations[0]
    keypoints0 = projection.project(X_true)
    keypoints1 = projection.project(transform(R, t, X_true))

    K = camera_parameters.matrix

    N = X_true.shape[0]
    for i in range(N):
        x_true = X_true[i]
        x, depth0, depth1 = linear_triangulation(
            keypoints0[i], keypoints1[i], R, t, K)
        assert_array_almost_equal(x, x_true)
        assert_equal(depth0, x[2])
        assert_equal(depth1, x[1] + t[2])


def test_extract_poses():
    def test(R_true, t_true):
        # skew matrx corresponding to t
        S_true = tangent_so3(t_true.reshape(1, *t_true.shape))[0]

        E_true = np.dot(R_true, S_true)

        R1, R2, t1, t2 = extract_poses(E_true)

        # t1 = -t2, R.T * t1 is parallel to t_true
        assert_array_almost_equal(t1, -t2)
        assert_array_almost_equal(np.cross(np.dot(R1.T, t1), t_true),
                                  np.zeros(3))
        assert_array_almost_equal(np.cross(np.dot(R2.T, t1), t_true),
                                  np.zeros(3))

        # make sure that both of R1 and R2 are rotation matrices
        assert_array_almost_equal(np.dot(R1.T, R1), np.identity(3))
        assert_array_almost_equal(np.dot(R2.T, R2), np.identity(3))
        assert_almost_equal(np.linalg.det(R1), 1.)
        assert_almost_equal(np.linalg.det(R2), 1.)


    for R, t in itertools.product(rotations, translations):
        test(R, t)
