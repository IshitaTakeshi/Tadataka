from pathlib import Path
import re

from vitamine.dataset.base import BaseDataset


def extract_timestamp(filename):
    return int(re.findall(r'\d+', filename)[0])


class CollaborativeDataset(BaseDataset):
    def load_paths(self):
        paths_color = sorted(list(self.dataset_root.glob("*.color.png")))
        paths_depth = sorted(list(self.dataset_root.glob("*.depth.png")))

        timestamps_color = []
        timestamps_depth = []
        for path_color, path_depth in zip(paths_color, paths_depth) :
            t_color = extract_timestamp(path_color.name)
            t_depth = extract_timestamp(path_depth.name)
            timestamps_color.append(t_color)
            timestamps_depth.append(t_depth)
        return timestamps_color, paths_color, timestamps_depth, paths_depth
