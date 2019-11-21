import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tadataka.rigid_transform import transform_all


def plot_cameras(ax, camera_rotations, camera_locations):
    ax.add_collection3d(
        cameras_poly3d(
            np.array(camera_rotations),
            np.array(camera_locations)
        )
    )
    return ax


def cameras_poly3d(camera_rotations, camera_locations, scale=1.0):
    # this code is the modified version of
    # [an answer](https://stackoverflow.com/a/44920709)
    # by [serenity](https://stackoverflow.com/users/2666859/serenity)

    v = np.array([
        [-0.5, -0.5, 1.5],
        [+0.5, -0.5, 1.5],
        [+0.5, +0.5, 1.5],
        [-0.5, +0.5, 1.5],
        [0, 0, 0]
    ])

    V = transform_all(camera_rotations, camera_locations, v * scale)

    verts = []
    for v in V:
        P = np.array([
            [v[0], v[1], v[4]],
            [v[0], v[3], v[4]],
            [v[2], v[1], v[4]],
            [v[2], v[3], v[4]],
        ])
        verts.append(P)
    verts = np.vstack(verts)

    return Poly3DCollection(verts, facecolors='cyan', linewidths=1,
                            edgecolors='r', alpha=.25)
