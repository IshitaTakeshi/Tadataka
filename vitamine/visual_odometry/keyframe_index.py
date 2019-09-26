
class KeyframeIndices(object):
    def __init__(self):
        self.latest_index = -1
        self.indices = []

    def __str__(self):
        return str(self.indices)

    def remove(self, i):
        del self.indices[i]

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def add_new(self):
        self.latest_index += 1
        self.indices.append(self.latest_index)
