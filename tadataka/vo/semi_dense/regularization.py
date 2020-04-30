import numpy as np
from scipy.signal import convolve2d
from tadataka.math import weighted_mean
from tadataka.numeric import safe_invert
from tadataka.vo.semi_dense.stat import is_statically_same


def create_mask_(inv_depth_map, variance):
    height, width = inv_depth_map.shape

    cy, cx = height // 2, width // 2
    mask = np.empty((height, width), dtype=np.int64)
    for y in range(height):
        for x in range(width):
            mask[y, x] = is_statically_same(
                inv_depth_map[cy, cx], inv_depth_map[y, x],
                variance, factor=2.0
            )
    return mask


def regularize_(D, W, v):
    mask = create_mask_(D, v).astype(np.float64)
    weights = W * mask
    return weighted_mean(D, weights)


def regularize(hypothesis, conv_size=3):
    assert(conv_size % 2 == 1)

    height, width = hypothesis.shape
    weight_map = safe_invert(hypothesis.variance_map)
    offset = conv_size // 2

    regularized = np.copy(hypothesis.inv_depth_map)
    for y in range(offset, regularized.shape[0]-offset):
        for x in range(offset, regularized.shape[1]-offset):
            ystart, yend = y-offset, y+offset+1
            xstart, xend = x-offset, x+offset+1
            regularized[y, x] = regularize_(
                hypothesis.inv_depth_map[ystart:yend, xstart:xend],
                weight_map[ystart:yend, xstart:xend],
                hypothesis.variance_map[y, x]
            )
    return regularized
