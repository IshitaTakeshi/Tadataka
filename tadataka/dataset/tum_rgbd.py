
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from skimage.io import imread

from tadataka.camera import CameraModel, CameraParameters, RadTan
from tadataka.dataset.frame import Frame
from tadataka.dataset.base import BaseDataset
from tadataka.dataset.tum import load_image_paths, synchronize
from tadataka.utils import value_list
from tadataka.pose import Pose


DEPTH_FACTOR = 5000.


def load_depth_image_paths(dataset_root):
    path = Path(dataset_root, "depth.txt")
    return load_image_paths(path, prefix=dataset_root)


def load_rgb_image_paths(dataset_root):
    path = Path(dataset_root, "rgb.txt")
    return load_image_paths(path, prefix=dataset_root)


def load_poses(path, delimiter=' '):
    array = np.loadtxt(path, delimiter=delimiter)
    timestamps = array[:, 0]
    positions = array[:, 1:4]
    quaternions = array[:, 4:8]
    rotations = Rotation.from_quat(quaternions)
    return timestamps, rotations, positions


def load_ground_truth_poses(dataset_root):
    return load_poses(Path(dataset_root, "groundtruth.txt"), delimiter=None)


def no_such_sequence_message(freiburg):
    return "No such sequence 'freiburg{}'".format(freiburg)


# TODO better to move these camera model definitions to text files
def get_camera_model_depth(freiburg):
    if freiburg == 1:
        return CameraModel(
            CameraParameters(focal_length=[591.1, 590.1],
                             offset=[331.0, 234.0]),
            RadTan([-0.0410, 0.3286, 0.0087, 0.0051, -0.5643])
        )
    if freiburg == 2:
        return CameraModel(
            CameraParameters(focal_length=[580.8, 581.8],
                             offset=[308.8, 253.0]),
            RadTan([-0.2297, 1.4766, 0.0005, -0.0075, -3.4194])
        )
    if freiburg == 3:
        return CameraModel(
            CameraParameters(focal_length=[567.6, 570.2],
                             offset=[324.7, 250.1]),
            RadTan([0, 0, 0, 0, 0])
        )
    raise ValueError(no_such_sequence_message(freiburg))


def get_camera_model_rgb(freiburg):
    if freiburg == 1:
        return CameraModel(
            CameraParameters(focal_length=[517.3, 516.5],
                             offset=[318.6, 255.3]),
            RadTan([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])
        )
    if freiburg == 2:
        return CameraModel(
            CameraParameters(focal_length=[520.9, 521.0],
                             offset=[325.1, 249.7]),
            RadTan([0.2312, -0.7849, -0.0033, -0.0001, 0.9172])
        )
    if freiburg == 3:
        return CameraModel(
            CameraParameters(focal_length=[535.4, 539.2],
                             offset=[320.1, 247.6]),
            RadTan([0, 0, 0, 0, 0])
        )
    raise ValueError(no_such_sequence_message(freiburg))


def get_depth_scale(freiburg):
    if freiburg == 1:
        return 1.035
    if freiburg == 2:
        return 1.031
    if freiburg == 3:
        return 1.000
    raise ValueError(no_such_sequence_message(freiburg))


# TODO download and set dataset_root and which_freiburg automatically
class TumRgbdDataset(BaseDataset):
    def __init__(self, dataset_root, which_freiburg):
        # there are 3 types of camera model
        # specify which to use by setting 'which_freiburg'

        self.depth_factor = DEPTH_FACTOR * get_depth_scale(which_freiburg)
        self.camera_model = get_camera_model_rgb(which_freiburg)
        self.camera_model_depth = get_camera_model_depth(which_freiburg)

        timestamps_gt, rotations, positions =\
            load_ground_truth_poses(dataset_root)

        timestamps_rgb, paths_rgb = load_rgb_image_paths(dataset_root)
        timestamps_depth, paths_depth = load_depth_image_paths(dataset_root)

        matches = synchronize(timestamps_gt, timestamps_rgb,
                              timestamps_ref=timestamps_depth)

        indices_gt = matches[:, 0]
        indices_rgb = matches[:, 1]
        indices_depth = matches[:, 2]

        self.length = matches.shape[0]

        self.timestamps = timestamps_gt[indices_gt]
        self.rotations = rotations[indices_gt]
        self.positions = positions[indices_gt]

        self.paths_rgb = value_list(paths_rgb, indices_rgb)
        self.paths_depth = value_list(paths_depth, indices_depth)

    def load(self, index):
        I = imread(self.paths_rgb[index])
        D = imread(self.paths_depth[index])
        D = D / self.depth_factor
        pose_world_camera = Pose(self.rotations[index], self.positions[index])
        return Frame(self.camera_model, pose_world_camera, I, D)
