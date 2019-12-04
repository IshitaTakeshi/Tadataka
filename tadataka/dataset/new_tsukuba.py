import csv
from pathlib import Path
from xml.etree import ElementTree as ET

from scipy.spatial.transform import Rotation
from skimage.io import imread
import numpy as np

from tadataka.dataset.frame import StereoFrame
from tadataka.dataset.base import BaseDataset


def load_depth(path):
    tree = ET.parse(path)
    root = tree.getroot()
    rows_node, cols_node, dt_node, data_node = root[0]
    height, width = int(rows_node.text), int(cols_node.text)

    depth_text = data_node.text
    depth_text = depth_text.replace('\n', '').strip()
    depth_map = np.fromstring(depth_text, sep=' ')
    return depth_map.reshape(height, width)


def load_poses(pose_path):
    poses = np.loadtxt(pose_path, delimiter=',')
    positions, euler_angles = poses[:, 0:3], poses[:, 3:6]
    rotations = Rotation.from_euler('xyz', euler_angles)
    return rotations, positions


def discard_alpha(image):
    return image[:, :, 0:3]


def calc_baseline_offset(rotation, baseline_length):
    local_offset = np.array([baseline_length, 0, 0])
    R = rotation.as_dcm()
    return np.dot(R, local_offset)


# TODO download and set dataset_root automatically
class NewTsukubaDataset(BaseDataset):
    def __init__(self, dataset_root):
        pose_path = Path(dataset_root, "camera_track.txt")

        self.baseline_length = 10.0
        self.rotations, self.positions = load_poses(pose_path)

        depth_dir = Path(dataset_root, "depth_maps")
        image_dir = Path(dataset_root, "images")

        self.depth_left_paths = sorted(Path(depth_dir, "left").glob("*.xml"))
        self.depth_right_paths = sorted(Path(depth_dir, "right").glob("*.xml"))
        self.image_left_paths = sorted(Path(image_dir, "left").glob("*.png"))
        self.image_right_paths = sorted(Path(image_dir, "right").glob("*.png"))

        assert((len(self.depth_left_paths) == len(self.depth_right_paths) ==
                len(self.image_left_paths) == len(self.image_right_paths)))

    def load(self, index):
        image_left = imread(self.image_left_paths[index])
        image_right = imread(self.image_right_paths[index])

        image_left = discard_alpha(image_left)
        image_right = discard_alpha(image_right)

        depth_left = load_depth(self.depth_left_paths[index])
        depth_right = load_depth(self.depth_right_paths[index])
        position_center = self.positions[index]

        rotation = self.rotations[index]
        offset = calc_baseline_offset(rotation, self.baseline_length)

        return StereoFrame(
            image_left, image_right,
            depth_left, depth_right,
            position_center - offset / 2.0,
            position_center + offset / 2.0,
            rotation
        )
