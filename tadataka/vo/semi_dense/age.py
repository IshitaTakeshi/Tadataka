import numpy as np
from tadataka.warp import Warp2D
from tadataka.coordinates import substitute, get
from tadataka.vo.semi_dense.coordinates import warp_coordinates


def increment_age(age_map0, warp10, depth_map0):
    assert(age_map0.shape == depth_map0.shape)
    us0, us1, _, _ = warp_coordinates(warp10, depth_map0)
    age_map1 = np.zeros(age_map0.shape, dtype=age_map0.dtype)
    return substitute(age_map1, us1.astype(np.int64), get(age_map0, us0) + 1)


class AgeMap(object):
    def __init__(self, camera_model0, pose0, depth_map0):
        self._f = camera_model0, pose0, depth_map0
        self.age_map = np.zeros(depth_map0.shape, dtype=np.int64)

    def increment(self, camera_model1, pose1, depth_map1):
        camera_model0, pose0, depth_map0 = self._f
        warp10 = Warp2D(camera_model0, camera_model1, pose0, pose1)
        self.age_map = increment_age(self.age_map, warp10, depth_map0)
        self._f = camera_model1, pose1, depth_map1

    def get(self):
        return self.age_map
