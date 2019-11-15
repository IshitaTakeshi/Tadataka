from vitamine.plot.common import axis3d
from vitamine.plot.visualizers import plot3d
from vitamine.plot.cameras import plot_cameras
from vitamine.so3 import rodrigues
from matplotlib import pyplot as plt


def plot_map(camera_omegas, camera_locations, points, color=None):
    ax = axis3d()
    plot3d(ax, points, color)
    plot_cameras(ax, rodrigues(camera_omegas), camera_locations)
    plt.show()
