from matplotlib import pyplot as plt

from tadataka.plot.common import axis3d
from tadataka.plot.visualizers import plot3d_
from tadataka.plot.cameras import plot_cameras_


def plot_map(poses, points, colors=None, camera_scale=1.0):
    ax = axis3d()
    plot3d_(ax, points, colors)
    plot_cameras_(ax, poses, camera_scale)
    plt.show()
