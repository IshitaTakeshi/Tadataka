from autograd import numpy as np

from matplotlib import pyplot as plt

from vitamine.camera import CameraParameters
from vitamine.dataset.points import donut
from vitamine.dataset.observations import (
    generate_observations, generate_translations
)
from vitamine.projection.projections import PerspectiveProjection
from vitamine.rigid.transformation import transform_all
from vitamine.rigid.coordinates import world_to_camera
from vitamine.rigid.rotation import rodrigues
from vitamine.observations import Observations
from vitamine.visualization.visual_odometry import VisualOdometryAnimation
from vitamine.visual_odometry.visual_odometry import VisualOdometry
from vitamine.visualization.visualizers import set_aspect_equal, plot3d
from vitamine.visualization.cameras import cameras_poly3d


def set_invisible(observations, masks):
    observations[~masks] = np.nan
    return observations


def plot_observations(observations):
    N = observations.shape[0]
    nrows = 2
    ncols = N // nrows if N % nrows == 0 else N // nrows + 1
    fig, axes = plt.subplots(ncols, nrows)
    for i, ax in enumerate(axes.flatten()):
        P = observations[i]
        ax.scatter(P[:, 0], P[:, 1])
    plt.show()


camera_parameters = CameraParameters(
    focal_length=[1., 1.],
    offset=[0., 0.]
)
projection = PerspectiveProjection(camera_parameters)

camera_rotations, camera_locations, points_true =\
    donut(inner_r=8, outer_r=12, height=5, point_density=24, n_viewpoints=13)
rotations_true, translations_true =\
    world_to_camera(camera_rotations, camera_locations)

observations, masks = generate_observations(
    rotations_true, translations_true, points_true, projection)
observations = set_invisible(observations, masks)

vo = VisualOdometry(
    Observations(observations, window_size=8),
    camera_parameters
)

print("running visual odometry")
vo.sequence()
exit(0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

set_aspect_equal(ax)
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])

animation = VisualOdometryAnimation(fig, ax, vo.sequence, interval=100)
animation.plot()
