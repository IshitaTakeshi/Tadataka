import numpy as np

from tadataka.vo.semi_dense.common import invert_depth


def calc_new_inv_depth_map(inv_depth_map, pose01_tz):
    depth_map = invert_depth(inv_depth_map) - pose01_tz
    return invert_depth(depth_map)


def propagate_inv_depth_map(inv_depth_map, new_inv_depth_map, variance_map,
                            uncertaintity_map):
    """
    Propagate inverse depth map from t0 to t1
    """
    # the equations corresponding to this part in the thesis may be wrong
    # eq. 14 should be
    #   d1^{-1}(d0^{-1}) = (d0 - tz)^{-1}
    # eq. 15 should be
    #   sigma_d1^2 = (d1^{-1} / d0^{-1})^4 * sigma_d0^2 + sigma_p^2

    variance_ratio = np.power(new_inv_depth_map / inv_depth_map, 4)
    return variance_ratio * variance_map + uncertaintity_map
