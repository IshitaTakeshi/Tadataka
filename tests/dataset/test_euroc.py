import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pathlib import Path

from scipy.spatial.transform import Rotation
from tadataka.dataset.euroc import EurocDataset
from tadataka.camera.parameters import CameraParameters
from tadataka.camera.distortion import RadTan


dataset_root = Path(Path(__file__).parent, "euroc",  "mav0")


def test_euroc():
    dataset = EurocDataset(dataset_root)
    frame_l, frame_r = dataset[0]

    camera_parameters = CameraParameters(
        focal_length=[42, 43],
        offset=[20, 15],
        image_shape=[40, 30]
    )
    distortion_model = RadTan([-0.20, 0.01, -0.001, -0.002])
    assert(frame_l.camera_model.camera_parameters == camera_parameters)
    assert(frame_l.camera_model.distortion_model == distortion_model)

    camera_parameters = CameraParameters(
        focal_length=[42, 44],
        offset=[20, 14],
        image_shape=[40, 30]
    )
    distortion_model = RadTan([-0.201, 0.001, -0.002, -0.001])
    assert(frame_r.camera_model.camera_parameters == camera_parameters)
    assert(frame_r.camera_model.distortion_model == distortion_model)

    for i, brightness in enumerate([0, 2, 4, 6, 8]):
        frame_l, _ = dataset[i]
        assert_array_equal(frame_l.image, brightness * np.ones((30, 40)))

    for i, brightness in enumerate([1, 3, 5, 7, 9]):
        _, frame_r = dataset[i]
        assert_array_equal(frame_r.image, brightness * np.ones((30, 40)))

    rotations_gt = Rotation.from_rotvec(np.arange(0.0, 1.5, 0.1).reshape(5, 3))
    positions = np.arange(0.0, 6.0, 0.2).reshape(10, 3)[::2]

    R_l = Rotation.from_rotvec([0, np.pi / 4, 0]).as_dcm()
    R_r = Rotation.from_rotvec([0, 0, np.pi / 4]).as_dcm()
    p_l = np.array([0.1, 0.2, 0.3])
    p_r = np.array([0.2, 0.4, 0.6])

    for i, (frame_l, frame_r) in enumerate(dataset):
        R = rotations_gt[i].as_dcm()
        assert_array_almost_equal(frame_l.rotation.as_dcm(), np.dot(R, R_l))
        assert_array_almost_equal(frame_r.rotation.as_dcm(), np.dot(R, R_r))
        assert_array_almost_equal(frame_l.position,
                                  positions[i] + np.dot(R, p_l))
        assert_array_almost_equal(frame_r.position,
                                  positions[i] + np.dot(R, p_r))
