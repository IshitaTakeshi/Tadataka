from pathlib import Path
import csv

from skimage.io import imread

from tadataka.dataset.frame import MonoFrame
from tadataka.dataset.base import BaseDataset


def load_paths(dataset_root):
    rgbd_path = Path(dataset_root, "rgbd.txt")
    with open(str(rgbd_path), "r") as f:
        reader = csv.reader(f, delimiter=' ')

        timestamps_rgb = []
        timestamps_depth = []
        image_paths = []
        depth_paths = []
        for row in reader:
            timestamps_rgb.append(float(row[0]))
            timestamps_depth.append(float(row[2]))

            path_color = str(Path(dataset_root, row[1]))
            path_depth = str(Path(dataset_root, row[3]))

            image_paths.append(path_color)
            depth_paths.append(path_depth)
    return timestamps_rgb, timestamps_depth, image_paths, depth_paths


# TODO download and set dataset_root automatically
class TUMDataset(BaseDataset):
    def __init__(self, dataset_root, depth_factor=5000.):
        self.timestamps_rgb, self.timestamps_depth,\
            self.image_paths, self.depth_paths = load_paths(dataset_root)
        self.depth_factor = depth_factor

    def load(self, index):
        I = imread(self.image_paths[index])
        D = imread(self.depth_paths[index])
        D = D / self.depth_factor

        # TODO load ground truth
        return MonoFrame(self.timestamps_rgb[index],
                         self.timestamps_depth[index], I, D)
