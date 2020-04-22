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
        d = yaml.load(f, Loader=yaml.FullLoader)

    intrinsics = np.array(d['intrinsics'])
    dist_coeffs = np.array(d['distortion_coefficients'])

    T_bs = np.array(d['T_BS']['data']).reshape(4, 4)

    return intrinsics, dist_coeffs, T_bs


def wxyz_to_xyzw(wxyz):
    return wxyz[:, [1, 2, 3, 0]]


def load_poses(path, delimiter=','):
    array = np.loadtxt(path, delimiter=delimiter)
    timestamps = array[:, 0]
    positions = array[:, 1:4]
    quaternions = wxyz_to_xyzw(array[:, 4:8])
    rotations = Rotation.from_quat(quaternions)
    return timestamps, rotations, positions


def load_body_poses(dataset_root):
    path = Path(dataset_root, "state_groundtruth_estimate0", "data.csv")
    return load_poses(path)


class EurocDataset(BaseDataset):
    def __init__(self, dataset_root):
        intrinsics0, dist_coeffs0, self.T_bc0 =\
            load_camera_params(dataset_root, 0)
        intrinsics1, dist_coeffs1, self.T_bc1 =\
            load_camera_params(dataset_root, 1)

        self.camera_model0 = CameraModel(
            CameraParameters(focal_length=intrinsics0[0:2],
                             offset=intrinsics0[2:4]),
            RadTan(dist_coeffs0)
        )
        self.camera_model1 = CameraModel(
            CameraParameters(focal_length=intrinsics1[0:2],
                             offset=intrinsics1[2:4]),
            RadTan(dist_coeffs1)
        )

        timestamps0, image_paths0 = load_image_paths(dataset_root, 0)
        timestamps1, image_paths1 = load_image_paths(dataset_root, 1)
        timestamps_body, rotations_wb, t_wb = load_body_poses(dataset_root)

        matches = tum.synchronize(timestamps_body, timestamps0,
                                  timestamps_ref=timestamps1)
        indices_wb = matches[:, 0]
        indices0 = matches[:, 1]
        indices1 = matches[:, 2]
        self.rotations_wb = value_list(rotations_wb, indices_wb)
        self.t_wb = value_list(t_wb, indices_wb)
        self.image_paths0 = value_list(image_paths0, indices0)
        self.image_paths1 = value_list(image_paths1, indices1)
        self.length = matches.shape[0]

    def load(self, index):
        T_wb = Pose(self.rotations_wb[index], self.t_wb[index]).T
        T_wc0 = np.dot(T_wb, self.T_bc0)
        T_wc1 = np.dot(T_wb, self.T_bc1)

        R_wc0, t_wc0 = get_rotation_translation(T_wc0)
        R_wc1, t_wc1 = get_rotation_translation(T_wc1)

        pose_wc0 = Pose(Rotation.from_matrix(R_wc0), t_wc0)
        pose_wc1 = Pose(Rotation.from_matrix(R_wc1), t_wc1)

        I0 = imread(self.image_paths0[index])
        I1 = imread(self.image_paths1[index])

        return (Frame(self.camera_model0, pose_wc0, I0, None),
                Frame(self.camera_model1, pose_wc1, I1, None))
