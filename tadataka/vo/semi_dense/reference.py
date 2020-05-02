class ReferenceSelector(object):
    def __init__(self, age_map, values):
        self.age_map = age_map
        self.values = values

    def __call__(self, u):
        x, y = u
        age = self.age_map[y, x]
        if age == 0:
            return None

        return self.values[-age]


def relative_(pose_wr, pose_wk):
    pose_rk = pose_wr.inv() * pose_wk
    return pose_rk


def make_reference_selector(age_map, refframes, pose_wk):
    fs = [(c, i, relative_(pose_wr, pose_wk).T) for c, i, pose_wr in refframes]
    return ReferenceSelector(age_map, fs)

