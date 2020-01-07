import numpy as np
from matplotlib import pyplot as plt

from tadataka.coordinates import local_to_world
from tadataka.plot.common import axis3d
from tadataka.plot.visualizers import plot3d_
from tadataka.plot.cameras import plot_cameras_


def plot_map(poses, points, colors=None):
    ax = axis3d()
    plot3d_(ax, points, colors)
    plot_cameras_(ax, poses)
    plt.show()
