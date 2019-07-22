from autograd import numpy as np

from vitamine.bundle_adjustment.triangulation import two_view_reconstruction
from vitamine.camera import CameraParameters
from vitamine.dataset.points import cubic_lattice, corridor
from vitamine.dataset.observations import (
    generate_observations, generate_translations
)
from vitamine.projection.projections import PerspectiveProjection
from vitamine.rigid.transformation import transform_all
from vitamine.rigid.rotation import rodrigues


def generate_poses(n_viewpoints):
    omegas = np.zeros((n_viewpoints, 3))

    translations = np.vstack((
        np.zeros(n_viewpoints),
        np.zeros(n_viewpoints),
        np.arange(0, n_viewpoints) - 1.5
    )).T

    return omegas, translations


def set_invisible(observations, masks):
    observations[~masks] = np.nan
    return observations


from vitamine.visualization.visual_odometry import VisualOdometryAnimation
from vitamine.visual_odometry.visual_odometry import VisualOdometry
from vitamine.visualization.visualizers import set_aspect_equal


def plot_observations(observations):
    N = observations.shape[0]
    nrows = 2
    ncols = N // nrows if N % nrows == 0 else N // nrows + 1
    fig, axes = plt.subplots(ncols, nrows)
    for i, ax in enumerate(axes.flatten()):
        P = observations[i]
        print("i : ", np.all(np.isnan(P)))
        ax.scatter(P[:, 0], P[:, 1])
    plt.show()


omegas_true, translations_true = generate_poses(n_viewpoints=8)


points_true = corridor(width=2, height=3, length=8)

n_points = points_true.shape[0]

camera_parameters = CameraParameters(
    focal_length=[1., 1.],
    offset=[0., 0.]
)
projection = PerspectiveProjection(camera_parameters)

observations, masks = generate_observations(
    rodrigues(omegas_true), translations_true, points_true, projection)
observations = set_invisible(observations, masks)

print("observations.shape", observations.shape)

vo = VisualOdometry(observations, camera_parameters, window_size=8)


from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
set_aspect_equal(ax)
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 30])

animation = VisualOdometryAnimation(fig, ax, vo.frames, interval=100)
animation.plot()

