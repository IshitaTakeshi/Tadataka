
class KeyframeIndices(object):
    def __init__(self):
        self.indices = []

    def __str__(self):
        return str(self.indices)

    def remove(self, i):
        frame_index = self.indices[i]
        del self.indices[i]
        return frame_index

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __getitem__(self, i):
        return self.indices[i]

    def get_next(self):
        if len(self.indices) == 0:
            return 0
        return self.indices[-1] + 1

    def add_new(self, index):
        assert(index not in self.indices)
        self.indices.append(index)
