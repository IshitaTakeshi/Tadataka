import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tadataka.rigid_transform import transform


vertices_ = np.array([
    [-0.5, -0.5, 1.0],
    [+0.5, -0.5, 1.0],
    [+0.5, +0.5, 1.0],
    [-0.5, +0.5, 1.0],
    [0, 0, 0]
])


optical_axis_ = np.array([[0, 0, 0], [0, 0, 1]])


def plot_cameras_(ax, poses, scale=1.0):
    for pose in poses:
        ax.add_collection3d(camera_poly3d(pose, scale))
        ax.plot(*optical_axis(pose, scale), c='red')
    return ax


def optical_axis(pose, scale):
    V = transform(pose.R, pose.t, optical_axis_ * scale)
    src, dst = V[0], V[1]
    return [[src[0], dst[0]], [src[1], dst[1]], [src[2], dst[2]]]


def camera_poly3d(pose, scale):
    # this code is the modified version of
    # [an answer](https://stackoverflow.com/a/44920709)
    # by [serenity](https://stackoverflow.com/users/2666859/serenity)

    v = transform(pose.R, pose.t, vertices_ * scale)

    P = np.array([
        [v[0], v[1], v[4]],
        [v[0], v[3], v[4]],
        [v[2], v[1], v[4]],
        [v[2], v[3], v[4]]
    ])

    return Poly3DCollection(P, facecolors='cyan', linewidths=1,
                            edgecolors='red', alpha=.25)
