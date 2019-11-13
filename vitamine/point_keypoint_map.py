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


def to_point_hashes(point_keypoint_map, keypoint_indices):
    point_hashes = []
    for index in keypoint_indices:
        try:
            h = point_keypoint_map[index]
        except KeyError:
            h = None
        point_hashes.append(h)
    return point_hashes


def triangulation_required(point_keypoint_map0, point_keypoint_map1, matches01):
    mask = np.zeros(len(matches01), dtype=np.bool)
    for i, (index0, index1) in enumerate(matches01):
        exists0 = keypoint_exists(point_keypoint_map0, index0)
        exists1 = keypoint_exists(point_keypoint_map1, index1)
        mask[i] = (not exists0) and (not exists1)
    return mask


def copy_required(src_map, dst_map, src_indices, dst_indices):
    assert(len(src_indices) == len(dst_indices))
    mask = np.zeros(len(src_indices), dtype=np.bool)
    for i, (src_index, dst_index) in enumerate(zip(src_indices, dst_indices)):
        src_exists = keypoint_exists(src_map, src_index)
        dst_exists = keypoint_exists(dst_map, dst_index)
        mask[i] = src_exists and not dst_exists
    return mask


def copy(src_map, dst_map, src_indices, dst_indices):
    for i, (src_index, dst_index) in enumerate(zip(src_indices, dst_indices)):
        point_hash = point_by_keypoint(src_map, src_index)
        dst_map[point_hash] = dst_index
    return dst_map


def copy_existing_points(point_keypoint_map0, point_keypoint_map1, matches01):
    # whicth match is processed
    # matches corresponding to mask == False are not triangulated
    for index0, index1 in matches01:
        exists0 = keypoint_exists(point_keypoint_map0, index0)
        exists1 = keypoint_exists(point_keypoint_map1, index1)

        if exists0 and exists1:
            warn_if_incorrect_match(point_keypoint_map0, point_keypoint_map1,
                                    index0, index1)
            continue

        if exists0:  # and not exists1
            point_hash = point_by_keypoint(point_keypoint_map0, index0)
            point_keypoint_map1[point_hash] = index1
            continue

        if exists1:  # and not exists0
            point_hash = point_by_keypoint(point_keypoint_map1, index1)
            point_keypoint_map0[point_hash] = index0
            continue

        raise ValueError(
            f"Neither keypoint index {index0} in viewpoint 0 "
            f" nor keypoint index {index1} in viewpoint 1 is subscribed"
        )
    return point_keypoint_map0, point_keypoint_map1


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
