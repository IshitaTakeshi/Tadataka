import itertools

from autograd import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_equal, assert_almost_equal)
from autograd.numpy.linalg import inv, norm

from vitamine.projection import PerspectiveProjection
from vitamine.camera import CameraParameters
from vitamine.so3 import tangent_so3, rodrigues
from vitamine.rigid_transform import transform, transform_all
from vitamine.triangulation import (
    estimate_fundamental, fundamental_to_essential, extract_poses,
    linear_triangulation, points_from_known_poses, pose_point_from_keypoints)


# TODO add the case such that x[3] = 0

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
    [[1 / np.sqrt(2), -1 / np.sqrt(2), 0],
     [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
     [0, 0, 1]],
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]],
])

translations = np.array([
    [-8, 4, 8],
    [-4, 8, 9],
    [-4, 1, 7]
])



def normalize(M):
    m = M.flatten()
    return M / (norm(m) * np.sign(m[-1]))


def test_estimate_fundamental():
    camera_parameters = CameraParameters(
        focal_length=[0.8, 1.2],
        offset=[0.8, 0.2]
    )
    projection = PerspectiveProjection(camera_parameters)

    R, t = rotations[0], translations[0]
    keypoints0 = projection.compute(X_true)
    keypoints1 = projection.compute(transform(R, t, X_true))

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
    camera_parameters = CameraParameters(
        focal_length=[1., 1.],
        offset=[0., 0.]
    )
    projection = PerspectiveProjection(camera_parameters)

    R0, t0 = rotations[0], translations[0]
    R1, t1 = rotations[1], translations[1]

    keypoints0 = projection.compute(transform(R0, t0, X_true))
    keypoints1 = projection.compute(transform(R1, t1, X_true))

    K = camera_parameters.matrix

    N = X_true.shape[0]
    for i in range(N):
        x_true = X_true[i]
        x, depth0, depth1 = linear_triangulation(
            R0, R1, t0, t1, keypoints0[i], keypoints1[i])
        assert_array_almost_equal(x, x_true)
        assert_equal(depth0, x[1] + t0[2])
        assert_equal(depth1, x[2] + t1[2])


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


def test_points_from_known_poses():
    camera_parameters = CameraParameters(
        focal_length=[1., 1.],
        offset=[0., 0.]
    )
    K = camera_parameters.matrix

    projection = PerspectiveProjection(camera_parameters)

    points = np.array([
        [-1, 3, 2],
        [1, 1, 1],
        [0, 4, 8]
    ])

    R = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])
    t = np.array([0, 1, 0])

    keypoints0 = projection.compute(points)
    keypoints1 = projection.compute(transform(R, t, points))
    # obviously points are in front of the both camers (depth > 0)
    X, valid_depth_mask = points_from_known_poses(
        np.identity(3), R, np.zeros(3), t,
        keypoints0, keypoints1
    )
    assert_array_almost_equal(
        projection.compute(X),
        keypoints0
    )
    assert_array_almost_equal(
        projection.compute(transform(R, t, points)),
        keypoints1
    )
    assert_equal(valid_depth_mask, np.array([1, 1, 1], dtype=np.bool))

    t = np.array([0, 0, -2])

    keypoints0 = projection.compute(points)
    keypoints1 = projection.compute(transform(R, t, points))
    # points[1] is behind the 2nd camera
    X, valid_depth_mask = points_from_known_poses(
        np.identity(3), R, np.zeros(3), t,
        keypoints0, keypoints1
    )

    assert_array_almost_equal(
        projection.compute(X),
        keypoints0
    )
    assert_array_almost_equal(
        projection.compute(transform(R, t, points)),
        keypoints1
    )
    assert_equal(valid_depth_mask, np.array([1, 0, 1], dtype=np.bool))


def test_pose_point_from_keypoints():
    camera_parameters = CameraParameters(
        focal_length=[1., 1.],
        offset=[0., 0.]
    )
    projection = PerspectiveProjection(camera_parameters)

    R0, t0 = rotations[0], translations[0]
    R1, t1 = rotations[1], translations[1]
    keypoints0 = projection.compute(transform(R0, t0, X_true))
    keypoints1 = projection.compute(transform(R1, t1, X_true))

    R, t, X, mask = pose_point_from_keypoints(keypoints0, keypoints1)

    # cannot evaluate pose (R, t) diretly
    # because the scale cannot be known

    assert_array_almost_equal(
        projection.compute(X),
        keypoints0
    )
    assert_array_almost_equal(
        projection.compute(transform(R, t, X)),
        keypoints1
    )
