import csv
import os
from pathlib import Path
from xml.etree import ElementTree as ET

from tqdm import tqdm
from scipy.spatial.transform import Rotation
from skimage.io import imread
import numpy as np

from tadataka.camera import CameraModel, CameraParameters, FOV
from tadataka.dataset.frame import Frame
from tadataka.dataset.base import BaseDataset
from tadataka.pose import Pose


def load_depth(path):
    tree = ET.parse(path)
    root = tree.getroot()
    rows_node, cols_node, dt_node, data_node = root[0]
    height, width = int(rows_node.text), int(cols_node.text)

    depth_text = data_node.text
    depth_text = depth_text.replace('\n', '').strip()
    depth_map = np.fromstring(depth_text, sep=' ')
    return depth_map.reshape(height, width)


def generate_cache(src_dir, cache_dir, src_extension, loader):
    def generate_(subdir):
        os.makedirs(str(Path(cache_dir, subdir)))

        print(f"Generating cache from {subdir}")

        paths = Path(src_dir, subdir).glob("*" + src_extension)
        for path in tqdm(list(paths)):
            filename = path.name.replace(src_extension, ".npy")
            cache_path = Path(cache_dir, subdir, filename)
            array = loader(path)
            np.save(str(cache_path), array)

    generate_("left")
    generate_("right")


def generate_image_cache(image_dir, cache_dir):
    print("Generating image cache")
    generate_cache(image_dir, cache_dir, ".png", imread)


def generate_depth_cache(depth_dir, cache_dir):
    print("Generating depth cache")
    generate_cache(depth_dir, cache_dir, ".xml", load_depth)


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
        depth_cache_dir = Path(groundtruth_dir, "depth_cache")

        if not depth_cache_dir.exists():
            generate_depth_cache(depth_dir, depth_cache_dir)

        self.depth_L_paths = sorted(Path(depth_cache_dir, "left").glob("*.npy"))
        self.depth_R_paths = sorted(Path(depth_cache_dir, "right").glob("*.npy"))

        image_dir = Path(illumination_dir, condition)
        image_cache_dir = Path(illumination_dir, condition + "_cache")

        if not image_cache_dir.exists():
            generate_image_cache(image_dir, image_cache_dir)

        self.image_L_paths = sorted(Path(image_cache_dir, "left").glob("*.npy"))
        self.image_R_paths = sorted(Path(image_cache_dir, "right").glob("*.npy"))

        assert((len(self.depth_L_paths) == len(self.depth_R_paths) ==
                len(self.image_L_paths) == len(self.image_R_paths) ==
                len(self.rotations) == len(self.positions)))

        for i in range(len(self.positions)):
            DL = self.depth_L_paths[i].name
            DR = self.depth_R_paths[i].name
            IL = self.image_L_paths[i].name
            IR = self.image_R_paths[i].name

            assert(DL[-8:] == DR[-8:] == IL[-8:] == IR[-8:])

    def __len__(self):
        return len(self.positions)

    def load(self, index):
        image_l = np.load(self.image_L_paths[index])
        image_r = np.load(self.image_R_paths[index])

        image_l = discard_alpha(image_l)
        image_r = discard_alpha(image_r)

        depth_l = np.load(self.depth_L_paths[index])
        depth_r = np.load(self.depth_R_paths[index])

        position_center = self.positions[index]
        rotation = self.rotations[index]

        offset = calc_baseline_offset(rotation, self.baseline_length)
        pose_wl = Pose(rotation, position_center - offset / 2.0)
        pose_wr = Pose(rotation, position_center + offset / 2.0)
        return (
            Frame(self.camera_model, pose_wl, image_l, depth_l),
            Frame(self.camera_model, pose_wr, image_r, depth_r)
        )
