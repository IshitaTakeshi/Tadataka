class BaseDataset(object):
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.load(index)

        # 'index' is a slice
        start = 0 if index.start is None else index.start
        stop = len(self) if index.stop is None else index.stop
        step = 1 if index.step is None else index.step
        return [self.load(i) for i in range(start, stop, step)]

    def __len__(self):
        return len(self.timestamps_rgb)
