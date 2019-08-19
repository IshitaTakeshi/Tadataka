from autograd import numpy as np

from vitamine.rigid.transformation import transform_all


local_hill = np.array([
    [1, 1, 1, 1, 1],
    [1, 2, 2, 2, 1],
    [1, 2, 3, 2, 1],
    [1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1],
])


def set_hills(curvature, coordinates):
    for x, y in coordinates:
        # fill around (x, y) with the hill
        curvature[y-2:y+3, x-2:x+3] = local_hill
    return curvature


def project(rotations, translations, points, projection):
    points = transform_all(rotations, translations, points)
    keypoints = projection.compute(points.reshape(-1, 3))
    keypoints = keypoints.reshape(points.shape[0], points.shape[1], 2)
    return keypoints
