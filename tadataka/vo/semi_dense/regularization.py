import numpy as np
from scipy.signal import convolve2d
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.vo.semi_dense.stat import is_statically_same
from numba import njit


@njit
def is_statically_same(inv_depth1, inv_depth2, variance, factor):
    # equivalent_condition:
    #   np.abs(inv_depth1-inv_depth2) <= factor * np.sqrt(variance1)
    ds = (inv_depth1 - inv_depth2) * (inv_depth1 - inv_depth2)
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


def regularize_(D, W, v):
    mask = create_mask_(D, v)
    M = W * mask.astype(np.float64)
    return (M * D).sum() / M.sum()


def regularize(inv_depth_map, variance_map, conv_size=3):
    assert(inv_depth_map.shape == variance_map.shape)
    height, width = inv_depth_map.shape
    weight_map = invert_depth(variance_map)
    offset = conv_size // 2

    regularized = np.copy(inv_depth_map)
    for y in range(offset, regularized.shape[0]-offset):
        for x in range(offset, regularized.shape[1]-offset):
            ystart, yend = y-offset, y+offset+1
            xstart, xend = x-offset, x+offset+1
            regularized[y, x] = regularize_(
                inv_depth_map[ystart:yend, xstart:xend],
                weight_map[ystart:yend, xstart:xend],
                variance_map[y, x]
            )
    return regularized
