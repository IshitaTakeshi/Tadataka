import numpy as np
from tadataka.vo.semi_dense.age import increment_age


class ReferenceFrameSelector(object):
    def __init__(self, frame0, inv_depth_map0):
        self.age_map = np.zeros(inv_depth_map0.shape, dtype=np.int64)
        self.inv_depth_map0 = inv_depth_map0
        self.frames = []
        self.frame0 = frame0

    def update(self, frame1, inv_depth_map1):
        self.age_map = increment_age(
            self.age_map, self.inv_depth_map0,
            self.frame0.camera_model, frame1.camera_model,
            (self.frame0.R, self.frame0.t), (frame1.R, frame1.t)
        )

        self.frames.append(self.frame0)

        self.inv_depth_map0 = inv_depth_map1
        self.frame0 = frame1

    def __call__(self, u):
        x, y = u
        age = self.age_map[y, x]
        if age == 0:
            return None
        return self.frames[-age]
