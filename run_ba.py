from autograd import numpy as np

from projection.projections import PerspectiveProjection
from camera import CameraParameters
from rigid.transformation import transform_each
from rigid.rotation import rodrigues
from bundle_adjustment.triangulation import two_view_reconstruction


def cubic_lattice(N):
    array = np.arange(N)
    xs, ys, zs = np.meshgrid(array, array, array)
    return np.vstack((xs.flatten(), ys.flatten(), zs.flatten())).T


def initialize(keypoints, camera_parameters):
    import cv2

    n_viewpoints = keypoints.shape[0]

    K = camera_parameters.matrix

    R, t, points = two_view_reconstruction(keypoints[0], keypoints[1], K)

    omegas = np.zeros((n_viewpoints, 3))
    translations = np.zeros((n_viewpoints, 3))
    for i in range(n_viewpoints):
        retval, rvec, tvec = cv2.solvePnP(points, keypoints[i], K, np.zeros(4))
        omegas[i] = rvec.flatten()
        translations[i] = tvec.flatten()
    return omegas, translations, points



def generate_translations(n_viewpoints, points, rotations):
    offset = 2.0
    translations = np.empty((n_viewpoints, 3))
    for i in range(n_viewpoints):
        R = rotations[i]
        # convert all ponits to the camera coordinates
        P = np.dot(R, points.T).T
        # search the point which has the minimum z value
        argmin = np.argmin(P[:, 2])
        p = P[argmin]
        translations[i] = -p + np.array([0, 0, offset])
        print("R * points + translation = P + translation = ", P - p)
        # print(R.dot(points.T).T + translations[i])
    return translations


def generate_observations(points, n_viewpoints, projection):
    n_points = points.shape[0]

    omegas = np.random.uniform(-1, 1, (n_viewpoints, 3))

    RS = rodrigues(omegas)
    translations = generate_translations(n_viewpoints, points, RS)

    points = transform_each(RS, translations, points)
    points = points.reshape(-1, 3)
    print(points[:, 2])
    assert((points[:, 2] >= 0).all())
    observations = projection.project(points)
    observations = observations.reshape(n_viewpoints, n_points, 2)

    return observations



camera_parameters = CameraParameters(
    focal_length=[1., 1.],
    offset=[0., 0.]
)


points_true = cubic_lattice(3)
n_viewpoints = 128
print(points_true.shape)
projection = PerspectiveProjection(camera_parameters)

observations = generate_observations(points_true, n_viewpoints, projection)

omegas, translations, points = initialize(observations, camera_parameters)
