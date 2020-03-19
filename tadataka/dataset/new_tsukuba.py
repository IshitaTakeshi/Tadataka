import csv
from pathlib import Path
from xml.etree import ElementTree as ET

from scipy.spatial.transform import Rotation
from skimage.io import imread
import numpy as np

from tadataka.camera import CameraModel, CameraParameters, FOV
from tadataka.dataset.frame import Frame
from tadataka.dataset.base import BaseDataset
from tadataka.pose import WorldPose


def load_depth(path):
    tree = ET.parse(path)
    root = tree.getroot()
    rows_node, cols_node, dt_node, data_node = root[0]
    height, width = int(rows_node.text), int(cols_node.text)

    depth_text = data_node.text
    depth_text = depth_text.replace('\n', '').strip()
    depth_map = np.fromstring(depth_text, sep=' ')
    return depth_map.reshape(height, width)


def align_coordinate_system(positions, euler_angles):
    # Camera coordinate system and world coordinate system are not aligned
    #
    # Usually camera coordinate system is represented in the format that
    # x: right  y: down  z: forward
    # however, in 'camera_track.txt', they are written in
    # x: right  y: up    z: backward
    #
    # This means the camera coordinate system is
    # rotated 180 degrees around the x-axis from the world coordinate system

    # rotate 180 degrees around the x-axis
    R = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    positions = np.dot(R, positions.T).T

    # Reverse rotations around y and z because axes are flipped
    # (rot_x, rot_y, rot_z) <- (rot_x, -rot_y, -rot_z)
    euler_angles[:, 1:3] = -euler_angles[:, 1:3]
    return positions, euler_angles


def load_poses(pose_path):
    poses = np.loadtxt(pose_path, delimiter=',')
    positions, euler_angles = poses[:, 0:3], poses[:, 3:6]
    positions, euler_angles = align_coordinate_system(positions, euler_angles)
    rotations = Rotation.from_euler('xyz', euler_angles, degrees=True)
    return rotations, positions


def discard_alpha(image):
    return image[:, :, 0:3]


def calc_baseline_offset(rotation, baseline_length):
    local_offset = np.array([baseline_length, 0, 0])
    R = rotation.as_matrix()
    return np.dot(R, local_offset)


# TODO download and set dataset_root automatically
class NewTsukubaDataset(BaseDataset):
    def __init__(self, dataset_root, condition="daylight"):

        self.camera_model = CameraModel(
            CameraParameters(focal_length=[615, 615], offset=[320, 240]),
            distortion_model=None
        )
        groundtruth_dir = Path(dataset_root, "groundtruth")
        illumination_dir = Path(dataset_root, "illumination")

        pose_path = Path(groundtruth_dir, "camera_track.txt")

        self.baseline_length = 10.0
        self.rotations, self.positions = load_poses(pose_path)

        depth_dir = Path(groundtruth_dir, "depth_maps")
        image_dir = Path(illumination_dir, condition)

        self.depth_L_paths = sorted(Path(depth_dir, "left").glob("*.xml"))
        self.depth_R_paths = sorted(Path(depth_dir, "right").glob("*.xml"))
        self.image_L_paths = sorted(Path(image_dir, "left").glob("*.png"))
        self.image_R_paths = sorted(Path(image_dir, "right").glob("*.png"))

        assert((len(self.depth_L_paths) == len(self.depth_R_paths) ==
                len(self.image_L_paths) == len(self.image_R_paths)))

        self.length = len(self.depth_L_paths)

    def load(self, index):
        image_l = imread(self.image_L_paths[index])
        image_r = imread(self.image_R_paths[index])

        image_l = discard_alpha(image_l)
        image_r = discard_alpha(image_r)

        depth_l = load_depth(self.depth_L_paths[index])
        depth_r = load_depth(self.depth_R_paths[index])

        position_center = self.positions[index]
        rotation = self.rotations[index]

        offset = calc_baseline_offset(rotation, self.baseline_length)
        pose_l = WorldPose(rotation, position_center - offset / 2.0)
        pose_r = WorldPose(rotation, position_center + offset / 2.0)
        return (
            Frame(self.camera_model, pose_l, image_l, depth_l),
            Frame(self.camera_model, pose_r, image_r, depth_r)
        )
