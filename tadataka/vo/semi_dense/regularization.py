import numpy as np
from scipy.signal import convolve2d
from tadataka.vo.semi_dense.common import invert_depth


def regularize(inv_depth_map, variance_map):
    weight_map = invert_depth(variance_map)
    R = convolve2d(inv_depth_map * weight_map, np.ones((3, 3)), mode="valid")
    W = convolve2d(weight_map, np.ones((3, 3)), mode="valid")
    return R / W
