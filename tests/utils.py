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


def unit_uniform(shape):
    return np.random.uniform(-1.0, 1.0, shape)


def add_uniform_noise(array, scale=0.01):
    noise = scale * unit_uniform(array.shape)
    return array + noise


def relative_error(x_true, x_pred):
    return np.linalg.norm(x_true - x_pred) / np.linalg.norm(x_true)


def random_rotation_matrix(size):
    A = np.random.random((size, size))
    return np.linalg.svd(np.dot(A.T, A))[0]


def random_binary(size):
    return np.random.randint(0, 2, size, dtype=np.bool)


def add_noise(descriptors, indices):
    descriptors = np.copy(descriptors)
    descriptors[indices] = random_binary((len(indices), descriptors.shape[1]))
    return descriptors


def break_other_than(descriptors, indices):
    indices_to_break = np.setxor1d(np.arange(len(descriptors)), indices)
    return add_noise(descriptors, indices_to_break)
