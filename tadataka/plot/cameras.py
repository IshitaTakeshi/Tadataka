import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tadataka.rigid_transform import transform


vertices = np.array([
    [-0.05, -0.05, 0.15],
    [+0.05, -0.05, 0.15],
    [+0.05, +0.05, 0.15],
    [-0.05, +0.05, 0.15],
    [0, 0, 0]
])


def plot_cameras_(ax, poses, scale=1.0):
    ax.add_collection3d(cameras_poly3d(poses, scale))
    return ax


def cameras_poly3d(poses, scale):
    # this code is the modified version of
    # [an answer](https://stackoverflow.com/a/44920709)
    # by [serenity](https://stackoverflow.com/users/2666859/serenity)

    V = []
    for pose in poses:
        pose = pose.local_to_world()
        v = transform(pose.rotation.as_matrix(), pose.t, vertices * scale)

        P = np.array([
            [v[0], v[1], v[4]],
            [v[0], v[3], v[4]],
            [v[2], v[1], v[4]],
            [v[2], v[3], v[4]],
        ])
        V.append(P)
    V = np.vstack(V)

    return Poly3DCollection(V, facecolors='cyan', linewidths=1,
                            edgecolors='r', alpha=.25)
