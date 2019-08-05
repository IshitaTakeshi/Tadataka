from pathlib import Path
import csv
from skimage.io import imread
from vitamine.dataset.frame import Frame


def load(rgbd_path):
    with open(str(rgbd_path), "r") as f:
        reader = csv.reader(f, delimiter=' ')

        timestamps_rgb = []
        paths_rgb = []
        timestamps_depth = []
        paths_depth = []
        for row in reader:
            timestamps_rgb.append(float(row[0]))
            paths_rgb.append(row[1])
            timestamps_depth.append(float(row[2]))
            paths_depth.append(row[3])
    return timestamps_rgb, paths_rgb, timestamps_depth, paths_depth


class TUMDataset(object):
    def __init__(self, dataset_root, depth_factor=5000):
        self.dataset_root = dataset_root
        self.depth_factor = depth_factor

        path = Path(dataset_root, "rgbd.txt")
        self.timestamps_rgb, self.paths_rgb,\
            self.timestamps_depth, self.paths_depth = load(path)

    def __len__(self):
        return len(self.timestamps_rgb)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.load(index)

        indices = range(index.start, index.stop, index.step)
        return [self.load(i) for i in indices]

    def load(self, index):
        timestamp_rgb = self.timestamps_rgb[index]
        timestamp_depth = self.timestamps_depth[index]

        path_rgb = str(Path(self.dataset_root, self.paths_rgb[index]))
        path_depth = str(Path(self.dataset_root, self.paths_depth[index]))

        I = imread(path_rgb)
        D = imread(path_depth)
        D = D / self.depth_factor

        return Frame(timestamp_rgb, timestamp_depth, I, D)
