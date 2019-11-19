import numpy as np

from tadataka.so3 import rodrigues


def cubic_lattice(N):
    array = np.arange(N)
    xs, ys, zs = np.meshgrid(array, array, array)
    return np.vstack((xs.flatten(), ys.flatten(), zs.flatten())).T


def donut(inner_r, outer_r, height=5, point_density=24, n_viewpoints=60,
          offset=1e-3):
    assert(isinstance(height, int))
    assert(outer_r > inner_r)

    # generate points on the xz-plane
    def round_points(thetas):
        return np.vstack([
            np.cos(thetas),
            np.zeros(thetas.shape[0]),
            np.sin(thetas)
        ]).T

    def rings(level_y):
        thetas = np.linspace(0, 2 * np.pi, point_density + 1)[:-1]
        inner = inner_r * round_points(thetas)
        outer = outer_r * round_points(thetas)
        inner[:, 1] = level_y
        outer[:, 1] = level_y
        return np.vstack((inner, outer))

    point_ys = np.arange(height)
    points = np.vstack([rings(level_y) for level_y in point_ys])

    camera_r = (inner_r + outer_r) / 2.
    camera_y = (point_ys[0] + point_ys[-1]) / 2.

    # add offset to avoid division by zero at projection
    thetas = np.linspace(0, 2 * np.pi, n_viewpoints + 1)[:-1] + offset
    camera_locations = camera_r * round_points(thetas)
    camera_locations[:, 1] = camera_y

    camera_omegas = np.vstack((
        np.zeros(n_viewpoints),
        -thetas,
        np.zeros(n_viewpoints)
    )).T

    return camera_omegas, camera_locations, points
