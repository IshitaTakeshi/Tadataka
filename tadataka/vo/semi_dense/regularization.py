import numpy as np
from scipy.signal import convolve2d
from tadataka.vo.semi_dense.common import invert_depth
from numba import njit


def are_statically_same(value1, value2, variance, factor):
    # equivalent_condition:
    #   np.abs(value1-value2) <= factor * np.sqrt(variance1)
    ds = (value1 - value2) * (value1 - value2)
    fs = factor * factor
    return ds <= fs * variance


def create_mask_(inv_depth_map, variance):
    height, width = inv_depth_map.shape

    cy, cx = height // 2, width // 2
    mask = np.empty((height, width), dtype=np.int64)
    for y in range(height):
        for x in range(width):
            mask[y, x] = are_statically_same(
                inv_depth_map[cy, cx], inv_depth_map[y, x],
                variance, factor=2.0
            )
    return mask


def regularize(inv_depth_map, variance_map, conv_size=3):
    assert(inv_depth_map.shape == variance_map.shape)
    height, width = inv_depth_map.shape
    weight_map = invert_depth(variance_map)
    offset = conv_size // 2

    regularized = np.empty((height-conv_size+1, width-conv_size+1))
    for y in range(regularized.shape[0]):
        for x in range(regularized.shape[1]):
            D = inv_depth_map[y:y+conv_size, x:x+conv_size]
            W = weight_map[y:y+conv_size, x:x+conv_size]
            mask = create_mask_(D, variance_map[y+offset, x+offset])
            M = W * mask.astype(np.float64)
            regularized[y, x] = (M * D).sum() / M.sum()
    return regularized
