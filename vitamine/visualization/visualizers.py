import numpy as np

from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def object_color(X):
    color = np.mean(np.abs(X), axis=1)
    return color / np.max(color)


def annotate(ax, P, labels=None):
    if labels is None:
        labels = range(len(P))

    font = FontProperties()
    font.set_weight("bold")

    for label, p in zip(labels, P):
        ax.text(*p, label, alpha=0.8, fontproperties=font)


def set_aspect_equal(ax):
    # This method is a modification of a work by Mateen Ulhaq,
    # which is distributed under the CC BY-SA 3.0 license
    # https://stackoverflow.com/a/50664367/2395994
    # https://creativecommons.org/licenses/by-sa/3.0/

    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    begin = origin - radius
    end = origin + radius

    ax.set_xlim3d([begin[0], end[0]])
    ax.set_ylim3d([begin[1], end[1]])
    ax.set_zlim3d([begin[2], end[2]])


def plot2d(P: np.ndarray, do_annotate=False, color=None):
    """
    Plot 2D points

    Args:
        P: 2D array of shape (n_points, 2)
        do_annotate: Annotate points if True
        color: Color of points
    """
    if color is None:
        color = object_color(P)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(P[:, 0], P[:, 1], c=color)

    if do_annotate:
        annotate(ax, P)

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_aspect('equal', 'datalim')


def plot3d(P: np.ndarray, do_annotate=False, color=None, elev=45, azim=0):
    """
    Plot 3D points

    Args:
        P: 3D array of shape (n_points, 3)
        do_annotate: Annotate points if True
        color: Color of points
        elev: Elevation of the viewpoint
        azim: Azimuth angle of the viewpoint
    """

    if color is None:
        color = object_color(P)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=color)

    if do_annotate:
        annotate(ax, P)

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev, azim)
    set_aspect_equal(ax)


def plot_result(X, viewpoints):
    # Define a color for each point

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    V = viewpoints
    ax.scatter(V[:, 0], V[:, 1], V[:, 2],
               c='r', marker='s', label='viewpoints')

    color = object_color(X)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color)

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.legend()

    set_aspect_equal(ax)
