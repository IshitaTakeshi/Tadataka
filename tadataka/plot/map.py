import numpy as np
from matplotlib import pyplot as plt

from tadataka.coordinates import local_to_world
from tadataka.plot.common import axis3d
from tadataka.plot.visualizers import plot3d_
from tadataka.plot.cameras import plot_cameras
from tadataka.so3 import rodrigues


def plot_map_(camera_omegas, camera_locations, points, color=None):
    ax = axis3d()
    plot3d_(ax, points, color)
    plot_cameras(ax, rodrigues(camera_omegas), camera_locations)
    plt.show()


def plot_map(poses, points, colors=None):
    omegas, translations = zip(*[[p.omega, p.t] for p in poses])
    omegas = np.array(omegas)
    translations = np.array(translations)
    omegas, translations = local_to_world(omegas, translations)
    plot_map_(omegas, translations, points, colors)
