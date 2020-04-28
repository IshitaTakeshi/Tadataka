import numpy as np
from tadataka.numeric import safe_invert
from tadataka.projection import inv_pi
from tadataka.matrix import get_rotation_translation
from tadataka.triangulation import calc_depth0_
from tadataka.rigid_transform import transform_se3


def calc_ref_inv_depth(T_rk, x_key, inv_depth_key):
    depth_key = safe_invert(inv_depth_key)
    p_key = inv_pi(x_key, depth_key)
    R_rk, t_rk = get_rotation_translation(T_rk)
    depth_ref = np.dot(R_rk[2], p_key) + t_rk[2]
    return safe_invert(depth_ref)


def calc_key_depth(T_rk, x_key, x_ref):
    R_rk, t_rk = get_rotation_translation(T_rk)
    return calc_depth0_(R_rk, t_rk, x_key, x_ref)


# TODO move this to a proper file
def clamp(value, min_, max_):
    return min(max(value, min_), max_)


class InvDepthSearchRange(object):
    def __init__(self, min_inv_depth, max_inv_depth, factor=2.0):
        assert(0 < min_inv_depth < max_inv_depth)
        self.factor = factor
        self.min_inv_depth = min_inv_depth
        self.max_inv_depth = max_inv_depth

    def _clamp(self, value):
        return clamp(value, self.min_inv_depth, self.max_inv_depth)

    def __call__(self, hypothesis):
        inv_depth, variance = hypothesis
        assert(variance >= 0.0)
        min_ = inv_depth - self.factor * variance
        max_ = inv_depth + self.factor * variance

        if max_ <= self.min_inv_depth or self.max_inv_depth <= min_:
            return None

        min_inv_depth = self._clamp(min_)
        max_inv_depth = self._clamp(max_)
        return min_inv_depth, max_inv_depth


def depth_search_range(min_inv_depth, max_inv_depth):
    min_depth = safe_invert(max_inv_depth)
    max_depth = safe_invert(min_inv_depth)
    return min_depth, max_depth
