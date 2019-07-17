from autograd import numpy as np

from vitamine.bundle_adjustment.triangulation import two_view_reconstruction
from vitamine.bundle_adjustment.bundle_adjustment import BundleAdjustment
from vitamine.camera import CameraParameters
from vitamine.dataset.points import cubic_lattice, corridor
from vitamine.dataset.observations import (
    generate_observations, generate_translations
)
from vitamine.projection.projections import PerspectiveProjection
from vitamine.rigid.transformation import transform_each
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


class VisualOdometry(object):
    def __init__(self, observations, window_size, start=0, end=None):
        n_observations = observations.shape[0]
        self.observations = observations
        self.window_size = window_size
        self.start = start
        self.end = n_observations if end is None else max(n_observations, end)
        assert(self.start < self.end)

    def frames(self):
        for i in range(self.start, self.end-self.window_size+1):
            yield self.estimate(i)

    def estimate(self, i):
        ba = BundleAdjustment(
            self.observations[i:i+self.window_size],
            camera_parameters,
            # initial_omegas=omegas,
            # initial_translations=translations,
            # initial_points=points
        )
        return ba.optimize()


from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from vitamine.visualization.visualizers import set_aspect_equal, object_color


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


class VisualOdometryAnimation(object):
    def __init__(self, fig, ax, frames, interval=100):
        self.ax = ax
        self.animation = FuncAnimation(fig, self.animate, frames=frames,
                                       interval=interval)

    def animate(self, args):
        omegas, translations, points = args
        return self.ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    def plot(self):
        plt.show()


omegas_true, translations_true = generate_poses(n_viewpoints=12)

points_true = corridor(width=2, height=4, length=2)
n_points = points_true.shape[0]

camera_parameters = CameraParameters(
    focal_length=[1., 1.],
    offset=[0., 0.]
)
projection = PerspectiveProjection(camera_parameters)

observations, masks = generate_observations(
    rodrigues(omegas_true), translations_true, points_true, projection)
observations = set_invisible(observations, masks)

vo = VisualOdometry(observations, window_size=8)


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

