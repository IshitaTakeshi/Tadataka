import numpy as np


def cubic_lattice(N):
    array = np.arange(N)
    xs, ys, zs = np.meshgrid(array, array, array)
    return np.vstack((xs.flatten(), ys.flatten(), zs.flatten())).T
