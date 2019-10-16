from autograd import numpy as np


def init_empty_points():
    return np.empty((0, 3), dtype=np.float64)


def get_point_indices(existing_points, new_points):
    start = len(existing_points)
    diff = len(new_points)
    return np.arange(start, start + diff)


def concat_points(points, new_points):
    point_indices = get_point_indices(points, new_points)
    points = np.vstack((points, new_points))
    return points, point_indices
