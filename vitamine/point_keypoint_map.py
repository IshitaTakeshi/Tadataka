import warnings
import numpy as np

from collections import defaultdict
from bidict import bidict


def init_correspondence(*args, **kwargs):
    return bidict(*args, **kwargs)


def point_by_keypoint(point_keypoint_map, keypoint_index):
    return point_keypoint_map.inverse[keypoint_index]


def point_exists(point_keypoint_map, keypoint_index):
    return keypoint_index in point_keypoint_map.values()


def associate_new(point_keypoint_map0, point_keypoint_map1,
                  point_hashes, matches01):
    assert(len(matches01) == len(point_hashes))
    for (index0, index1), point_hash in zip(matches01, point_hashes):
        point_keypoint_map0[point_hash] = index0
        point_keypoint_map1[point_hash] = index1
    return point_keypoint_map0, point_keypoint_map1


def warn_if_incorrect_match(point_keypoint_map0, point_keypoint_map1,
                            keypoint_index0, keypoint_index1):
    point_hash0 = point_by_keypoint(point_keypoint_map0, keypoint_index0)
    point_hash1 = point_by_keypoint(point_keypoint_map1, keypoint_index1)
    if point_hash0 != point_hash1:
        message = (
            "Wrong match! "
            "Keypoint index {} in viewpoint 0 and "
            "keypoint index {} in viewpoint 1 have matched, "
            "but indicate different 3D points!"
        )
        warnings.warn(message.format(keypoint_index0, keypoint_index1),
                      RuntimeWarning)


def triangulation_required(map_, keypoint_indices):
    return np.array([not point_exists(map_, i) for i in keypoint_indices])


def copy_required(map_, keypoint_indices):
    return np.array([point_exists(map_, i) for i in keypoint_indices])


def get_point_hashes(map_, keypoint_indices):
    return [point_by_keypoint(map_, i) for i in keypoint_indices]


def get_correspondences(correspondences, matches):
    assert(len(matches) == len(correspondences))
    point_hashes = []
    keypoint_indices = []
    for map_, matches01 in zip(correspondences, matches):
        for index0, index1 in matches01:
            try:
                point_hash = point_by_keypoint(map_, index0)
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
