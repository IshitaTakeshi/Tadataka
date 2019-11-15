
from pathlib import Path

from tadataka.dataset.frame import Frame
from skimage.io import imread


class BaseDataset(object):
    def __init__(self, dataset_root, depth_factor=1.):
        self.dataset_root = dataset_root
        self.depth_factor = depth_factor

        self.timestamps_color, self.paths_color,\
            self.timestamps_depth, self.paths_depth = self.load_paths()

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.load(index)

        start = 0 if index.start is None else index.start
        stop = len(self) if index.stop is None else index.stop
        step = 1 if index.step is None else index.step
        return [self.load(i) for i in range(start, stop, step)]

    def __len__(self):
        return len(self.paths_color)

    def load(self, index):
        I = imread(self.paths_color[index])
        D = imread(self.paths_depth[index])
        D = D / self.depth_factor

        return Frame(self.timestamps_color[index],
                     self.timestamps_depth[index],
                     I, D)
