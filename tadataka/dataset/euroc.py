import numpy as np
from pathlib import Path
import yaml
from skimage.io import imread
from scipy.spatial.transform import Rotation

from tadataka.dataset import tum
from tadataka.dataset.base import BaseDataset
from tadataka.dataset.frame import Frame
from tadataka.utils import value_list
from tadataka.camera.distortion import RadTan
from tadataka.camera.parameters import CameraParameters
from tadataka.camera.model import CameraModel
from tadataka.matrix import get_rotation_translation, motion_matrix
from tadataka.pose import Pose


def camera_dir(dataset_root, camera_index):
    return Path(dataset_root, "cam" + str(camera_index))


def load_image_paths(dataset_root, camera_index):
    d = camera_dir(dataset_root, camera_index)
    return tum.load_image_paths(Path(d, "data.csv"), Path(d, "data"),
                                delimiter=',')


def load_camera_params(dataset_root, camera_index):
    """
    EuRoC has 2 cameras. 'camera_index <- {0, 1}' specifies which to use
    """

    path = Path(camera_dir(dataset_root, camera_index), "sensor.yaml")

    with open(path, 'r') as f:
        d = yaml.load(f)

    resolution = d['resolution']
    intrinsics = np.array(d['intrinsics'])
    dist_coeffs = np.array(d['distortion_coefficients'])

    T = np.array(d['T_BS']['data']).reshape(4, 4)

    return resolution, intrinsics, dist_coeffs, T


def wxyz_to_xyzw(wxyz):
    return wxyz[:, [1, 2, 3, 0]]


def load_poses(path, delimiter=','):
    array = np.loadtxt(path, delimiter=delimiter)
    timestamps = array[:, 0]
    positions = array[:, 1:4]
    quaternions = wxyz_to_xyzw(array[:, 4:8])
    rotations = Rotation.from_quat(quaternions)
    return timestamps, rotations, positions


def load_ground_truth(dataset_root):
    path = Path(dataset_root, "state_groundtruth_estimate0", "data.csv")
    return load_poses(path)


class EurocDataset(BaseDataset):
    def __init__(self, dataset_root):
        [image0_w, image0_h], intrinsics0, dist_coeffs0, self.T0 =\
            load_camera_params(dataset_root, 0)
        [image1_w, image1_h], intrinsics1, dist_coeffs1, self.T1 =\
            load_camera_params(dataset_root, 1)

        self.camera_model0 = CameraModel(
            CameraParameters(focal_length=intrinsics0[0:2],
                             offset=intrinsics0[2:4],
                             image_shape=[image0_h, image0_w]),
            RadTan(dist_coeffs0)
        )
        self.camera_model1 = CameraModel(
            CameraParameters(focal_length=intrinsics1[0:2],
                             offset=intrinsics1[2:4],
                             image_shape=[image1_h, image1_w]),
            RadTan(dist_coeffs1)
        )

        timestamps0, image_paths0 = load_image_paths(dataset_root, 0)
        timestamps1, image_paths1 = load_image_paths(dataset_root, 1)
        timestamps_gt, rotations, positions = load_ground_truth(dataset_root)

        matches = tum.synchronize(timestamps_gt, timestamps0,
                                  timestamps_ref=timestamps1)
        indices_gt = matches[:, 0]
        indices0 = matches[:, 1]
        indices1 = matches[:, 2]
        self.rotations = value_list(rotations, indices_gt)
        self.positions = value_list(positions, indices_gt)
        self.image_paths0 = value_list(image_paths0, indices0)
        self.image_paths1 = value_list(image_paths1, indices1)
        self.length = matches.shape[0]

    def load(self, index):
        T = motion_matrix(self.rotations[index].as_dcm(), self.positions[index])
        R0, position0 = get_rotation_translation(np.dot(T, self.T0))
        R1, position1 = get_rotation_translation(np.dot(T, self.T1))

        pose0 = Pose(Rotation.from_dcm(R0), position0)
        pose1 = Pose(Rotation.from_dcm(R1), position1)

        I0 = imread(self.image_paths0[index])
        I1 = imread(self.image_paths1[index])

        return (Frame(self.camera_model0, pose0, I0, None),
                Frame(self.camera_model1, pose1, I1, None))
