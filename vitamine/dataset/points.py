import numpy as np

from vitamine.rigid.rotation import rodrigues


def cubic_lattice(N):
    array = np.arange(N)
    xs, ys, zs = np.meshgrid(array, array, array)
    return np.vstack((xs.flatten(), ys.flatten(), zs.flatten())).T


def straight_corridor(width, height, length):
    # they are just 'length's so the number of points are +1
    NX, NY, NZ = width + 1, height + 1, length + 1

    pillar = np.arange(0, NY)
    beam = np.arange(0, NX)

    xs = np.concatenate((
        beam,
        beam,
        beam[0] * np.ones(NY),
        beam[-1] * np.ones(NY)
    ))
    xs = np.tile(xs, NZ)
    xs = xs - width / 2.  # shift to make mean zero

    ys = np.concatenate((
        pillar[0] * np.ones(NX),
        pillar[-1] * np.ones(NX),
        pillar,
        pillar
    ))
    ys = np.tile(ys, NZ)
    ys = ys - height / 2. # shift to make mean zero

    zs = np.repeat(np.arange(NZ), NX * 2 + NY * 2)

    points = np.vstack((xs, ys, zs)).T

    camera_locations = np.vstack((
        np.zeros(length),
        np.zeros(length),
        np.arange(0, length) - 0.5
    )).T

    camera_rotations = rodrigues(np.zeros((length, 3)))
    return


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

    camera_rotations = rodrigues(
        np.vstack((np.zeros(n_viewpoints), -thetas, np.zeros(n_viewpoints))).T
    )

    return camera_rotations, camera_locations, points
