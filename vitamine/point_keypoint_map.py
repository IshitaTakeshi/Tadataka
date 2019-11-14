import warnings
import numpy as np

from collections import defaultdict
from bidict import bidict


def init_point_keypoint_map():
    return bidict()


def point_by_keypoint(point_keypoint_map, keypoint_index):
    return point_keypoint_map.inverse[keypoint_index]


def keypoint_exists(point_keypoint_map, keypoint_index):
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


def triangulation_required(point_keypoint_map0, point_keypoint_map1, matches01):
    mask = np.zeros(len(matches01), dtype=np.bool)
    for i, (index0, index1) in enumerate(matches01):
        exists0 = keypoint_exists(point_keypoint_map0, index0)
        exists1 = keypoint_exists(point_keypoint_map1, index1)
        mask[i] = (not exists0) and (not exists1)
    return mask


def copy_required(src_map, src_indices):
    mask = np.zeros(len(src_indices), dtype=np.bool)
    for i, src_index in enumerate(src_indices):
        mask[i] = keypoint_exists(src_map, src_index)
    return mask


def accumulate_shareable(point_keypoint_map, keypoint_indices):
    return [point_by_keypoint(point_keypoint_map, i) for i in keypoint_indices]


def correspondences(point_keypoint_maps, matches):
    assert(len(matches) == len(point_keypoint_maps))
    point_hashes = []
    keypoint_indices = []
    for map_, matches01 in zip(point_keypoint_maps, matches):
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


def merge_point_keypoint_maps(*maps):
    def update(M, map_):
        for key, value in map_.items():
            # avoid value duplication
            if (key not in M.keys()) and (value not in M.values()):
                M[key] = value
        return M

    M = init_point_keypoint_map()
    for map_ in maps:
        M = update(M, map_)
    return M
