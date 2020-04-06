class ReferenceSelector(object):
    def __init__(self, frames, age_map):
        self.frames = frames
        self.age_map = age_map

    def __call__(self, u):
        x, y = u
        age = self.age_map[y, x]
        if age == 0:
            return None
        return self.frames[-age]
