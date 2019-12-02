import csv
from pathlib import Path
from xml.etree import ElementTree as ET

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


def text_to_pose(line):
    pose = np.fromstring(line, sep=' ')
    position, rotvec = pose[0:3], pose[3:6]
    return rotvec, position


def load_poses(pose_path):
    rotvecs = []
    positions = []
    with open(pose_path) as f:
        for i, line in enumerate(f):
            rotvec, position = text_to_pose(line)
            rotvecs.append(rotvec)
            positions.append(position)
    return rotvecs, positions


def discard_alpha(image):
    return image[:, :, 0:3]


FPS = 30.


def load_timestamps(n_frames):
    return np.arange(0, n_frames) / FPS


# TODO download and set dataset_root automatically
class NewTsukubaDataset(BaseDataset):
    def __init__(self, dataset_root):
        pose_path = Path(dataset_root, "camera_track.txt")
        self.rotvecs, self.positions = load_poses(pose_path)

        N = len(self.rotvecs)

        timestamps = load_timestamps(N)
        self.timestamps_rgb = self.timestamps_depth = timestamps

        depth_dir = Path(dataset_root, "depth_maps")
        image_dir = Path(dataset_root, "images")

        self.depth_left_paths = sorted(Path(depth_dir, "left").glob("*.xml"))
        self.depth_right_paths = sorted(Path(depth_dir, "right").glob("*.xml"))
        self.image_left_paths = sorted(Path(image_dir, "left").glob("*.png"))
        self.image_right_paths = sorted(Path(image_dir, "right").glob("*.png"))

        assert(len(self.depth_left_paths) == len(self.depth_right_paths) == N)
        assert(len(self.image_left_paths) == len(self.image_right_paths) == N)

    def load(self, index):
        image_left = imread(self.image_left_paths[index])
        image_right = imread(self.image_right_paths[index])

        image_left = discard_alpha(image_left)
        image_right = discard_alpha(image_right)

        depth_left = load_depth(self.depth_left_paths[index])
        depth_right = load_depth(self.depth_right_paths[index])

        return StereoFrame(
            self.timestamps_rgb[index], self.timestamps_depth[index],
            image_left, image_right,
            depth_left, depth_right,
            self.rotvecs[index], self.positions[index]
        )
