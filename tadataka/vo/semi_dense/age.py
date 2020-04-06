import numpy as np
from tadataka.vo.semi_dense.coordinates import substitute, get, coordinates


def increment_age(age_map0, warp10, inv_depth_map0):
    assert(age_map0.shape == inv_depth_map0.shape)
    us0, us1, _, _ = coordinates(warp10, inv_depth_map0)
    age_map1 = np.zeros(age_map0.shape, dtype=age_map0.dtype)
    return substitute(age_map1, us1.astype(np.int64), get(age_map0, us0) + 1)
