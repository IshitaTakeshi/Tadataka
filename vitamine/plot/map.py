from vitamine.plot.visualizers import plot3d, axis3d
from vitamine.plot.cameras import plot_cameras
from vitamine.so3 import rodrigues
from matplotlib import pyplot as plt


def plot_map(camera_omegas, camera_locations, points):
    ax = axis3d()
    plot3d(ax, points)
    plot_cameras(ax, rodrigues(camera_omegas), camera_locations)
    plt.show()
