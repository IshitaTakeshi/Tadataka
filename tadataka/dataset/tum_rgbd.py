from pathlib import Path
import csv

import numpy as np
from skimage.io import imread
from scipy.spatial.transform import Rotation

from tadataka.dataset.frame import MonoFrame
from tadataka.dataset.base import BaseDataset
from tadataka.dataset.match import match_timestamps
from tadataka.utils import value_list


def load_image_paths(dataset_root, filepath):
    timestamps = []
    image_paths = []

    with open(str(filepath), "r") as f:
        reader = csv.reader(f, delimiter=' ')

        for row in reader:
            timestamps.append(float(row[0]))
            filepath = str(Path(dataset_root, row[1]))
            image_paths.append(filepath)
    return np.array(timestamps), image_paths


def load_depth_image_paths(dataset_root):
    return load_image_paths(dataset_root, Path(dataset_root, "depth.txt"))


def load_rgb_image_paths(dataset_root):
    return load_image_paths(dataset_root, Path(dataset_root, "rgb.txt"))


def load_poses(path):
    array = np.loadtxt(path)
    timestamps = array[:, 0]
    positions = array[:, 1:4]
    quaternions = array[:, 4:8]
    rotvecs = Rotation.from_quat(quaternions).as_rotvec()
    return timestamps, rotvecs, positions


def load_ground_truth_poses(dataset_root):
    return load_poses(Path(dataset_root, "groundtruth.txt"))


def syncronize(timestamps0, timestamps1, timestamps2, max_difference=0.02):
    matches01 = match_timestamps(timestamps0, timestamps1, max_difference)
    matches02 = match_timestamps(timestamps0, timestamps2, max_difference)
    indices0, indices1, indices2 = np.intersect1d(
        matches01[:, 0], matches02[:, 0], return_indices=True
    )
    return np.column_stack((indices0,
                            matches01[indices1, 1],
                            matches02[indices2, 1]))


# TODO download and set dataset_root automatically
class TUMDataset(BaseDataset):
    def __init__(self, dataset_root, depth_factor=5000.):
        self.depth_factor = depth_factor

        timestamps_gt, rotvecs, positions = load_ground_truth_poses(dataset_root)
        timestamps_rgb, paths_rgb = load_rgb_image_paths(dataset_root)
        timestamps_depth, paths_depth = load_depth_image_paths(dataset_root)

        matches = syncronize(timestamps_gt, timestamps_rgb, timestamps_depth)
        indices_gt = matches[:, 0]
        indices_rgb = matches[:, 1]
        indices_depth = matches[:, 2]

        self.rotvecs, self.positions = rotvecs[indices_gt], positions[indices_gt]
        self.paths_rgb = value_list(paths_rgb, indices_rgb)
        self.paths_depth = value_list(paths_depth, indices_depth)

    def load(self, index):
        I = imread(self.paths_rgb[index])
        D = imread(self.paths_depth[index])
        D = D / self.depth_factor

        # TODO load ground truth
        return MonoFrame(I, D, self.rotvecs[index], self.positions[index])
