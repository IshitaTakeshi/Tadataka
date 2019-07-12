import numpy as np


def cubic_lattice(N):
    array = np.arange(N)
    xs, ys, zs = np.meshgrid(array, array, array)
    return np.vstack((xs.flatten(), ys.flatten(), zs.flatten())).T


def corridor(width, height, length):
    # they are just 'length's so the number of points are +1
    NX = width + 1
    NY = height + 1
    NZ = length + 1

    pillar = np.arange(0, NY)
    beam = np.arange(0, NX)

    xs = np.concatenate((
        beam,
        beam,
        beam[0] * np.ones(NY),
        beam[-1] * np.ones(NY)
    ))
    xs = np.tile(xs, NZ)
    xs = xs - width / 2.

    ys = np.concatenate((
        pillar[0] * np.ones(NX),
        pillar[-1] * np.ones(NX),
        pillar,
        pillar
    ))
    ys = np.tile(ys, NZ)
    ys = ys - height / 2.

    zs = np.repeat(np.arange(NZ), NX * 2 + NY * 2)
    return np.vstack((xs, ys, zs)).T
