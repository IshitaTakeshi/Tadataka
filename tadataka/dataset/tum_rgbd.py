
from pathlib import Path

import numpy as np
from skimage.io import imread

from tadataka.camera import CameraModel, CameraParameters, FOV
from tadataka.dataset.frame import MonoFrame
from tadataka.dataset.base import BaseDataset
from tadataka.dataset.tum import load_poses, load_image_paths, synchronize
from tadataka.utils import value_list


def load_depth_image_paths(dataset_root):
    path = Path(dataset_root, "depth.txt")
    return load_image_paths(path, prefix=dataset_root)


def load_rgb_image_paths(dataset_root):
    path = Path(dataset_root, "rgb.txt")
    return load_image_paths(path, prefix=dataset_root)


def load_ground_truth_poses(dataset_root):
    return load_poses(Path(dataset_root, "groundtruth.txt"))


# TODO download and set dataset_root automatically
class TumRgbdDataset(BaseDataset):
    def __init__(self, dataset_root, depth_factor=5000.):
        self.depth_factor = depth_factor
        self.camera_model = CameraModel(
            CameraParameters(focal_length=[525., 525.], offset=[319.5, 239.5]),
            FOV(0.0)
        )

        timestamps_gt, rotations, positions =\
            load_ground_truth_poses(dataset_root)

        timestamps_rgb, paths_rgb = load_rgb_image_paths(dataset_root)
        timestamps_depth, paths_depth = load_depth_image_paths(dataset_root)

        matches = synchronize(timestamps_gt, timestamps_rgb, timestamps_depth)
        indices_gt = matches[:, 0]
        indices_rgb = matches[:, 1]
        indices_depth = matches[:, 2]

        self.rotations = rotations[indices_gt]
        self.positions = positions[indices_gt]

        self.paths_rgb = value_list(paths_rgb, indices_rgb)
        self.paths_depth = value_list(paths_depth, indices_depth)

    def load(self, index):
        I = imread(self.paths_rgb[index])
        D = imread(self.paths_depth[index])
        D = D / self.depth_factor

        return MonoFrame(self.camera_model, I, D,
                         self.positions[index], self.rotations[index])
