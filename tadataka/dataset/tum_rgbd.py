from pathlib import Path
import csv

from tadataka.dataset.base import BaseDataset


class TUMDataset(BaseDataset):
    def __init__(self, dataset_root):
        super().__init__(dataset_root, depth_factor=5000)

    def load_paths(self):
        rgbd_path = Path(self.dataset_root, "rgbd.txt")
        with open(str(rgbd_path), "r") as f:
            reader = csv.reader(f, delimiter=' ')

            timestamps_color = []
            paths_color = []
            timestamps_depth = []
            paths_depth = []
            for row in reader:
                path_color = str(Path(self.dataset_root, row[1]))
                path_depth = str(Path(self.dataset_root, row[3]))

                timestamps_color.append(float(row[0]))
                paths_color.append(path_color)
                timestamps_depth.append(float(row[2]))
                paths_depth.append(path_depth)
        return timestamps_color, paths_color, timestamps_depth, paths_depth
