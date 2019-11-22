import warnings
import numpy as np

from collections import defaultdict
from bidict import bidict

from tadataka.random import random_bytes


def init_correspondence(*args, **kwargs):
    return bidict(*args, **kwargs)


def point_by_keypoint(point_keypoint_map, keypoint_index):
    return point_keypoint_map.inverse[keypoint_index]


def point_exists(point_keypoint_map, keypoint_index):
    return keypoint_index in point_keypoint_map.values()


def get_point_hashes(map_, keypoint_indices):
    return [point_by_keypoint(map_, i) for i in keypoint_indices]


def get_indices(correspondence, matches01):
    point_hashes = []
    keypoint_indices = []
    for index0, index1 in matches01:
        try:
            point_hash = point_by_keypoint(correspondence, index0)
        except KeyError as e:
            # keypoint corresponding to 'index0' is not
            # triangulated yet
            continue

        point_hashes.append(point_hash)
        keypoint_indices.append(index1)
    return point_hashes, keypoint_indices


def merge_correspondences(*maps):
    M = init_correspondence()
    for map_ in maps:
        M.update(map_)
    return M


def generate_hashes(n_hashes, n_bytes=18):
    return [random_bytes(n_bytes) for i in range(n_hashes)]


def subscribe(point_array, matches01):
    assert(len(point_array) == len(matches01))

    point_hashes = generate_hashes(len(point_array))
    map0 = init_correspondence(zip(point_hashes, matches01[:, 0]))
    map1 = init_correspondence(zip(point_hashes, matches01[:, 1]))
    point_dict = dict(zip(point_hashes, point_array))
    return point_dict, map0, map1


def is_triangulated(correspondence, indices):
    return np.array([point_exists(correspondence, i) for i in indices])


def associate_triangulated(correspondence0, matches01):
    # Find keypoints that have corresponding 3D points
    # If keypoint in one frame has corresponding 3D point,
    # associate it to the matched keypoint in the other frame
    point_hashes0 = get_point_hashes(correspondence0, matches01[:, 0])
    return init_correspondence(zip(point_hashes0, matches01[:, 1]))
