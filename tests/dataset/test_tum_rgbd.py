from pathlib import Path

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
from scipy.spatial.transform import Rotation

from tadataka.dataset.tum_rgbd import TumRgbdDataset


dataset_root = Path(Path(__file__).parent, "tum_rgbd")


def test_tum_dataset():
    # 3rd frame should not be loaded because
    # the depth timestamp cannot match the corresponding pose timestamp

    valid_indices = [0, 1, 2, 4, 5, 6]
    dataset = TumRgbdDataset(dataset_root)
    image_shape = (30, 40)

    # test index access
    assert_equal(len(dataset), len(valid_indices))
    assert_equal(len(dataset[1:4:2]), 2)

    frame = dataset[0]
    assert_equal(frame.image.shape[0:2], image_shape)
    assert_equal(frame.depth_map.shape[0:2], image_shape)
    assert_equal(frame.image.shape[2], 3)

    angles = np.repeat(np.arange(0., 0.7, 0.1), 3).reshape(7, 3)
    expected_euler_angles = angles[valid_indices]
    expected_positions = np.arange(0., 0.21, 0.01).reshape(7, 3)
    expected_positions = expected_positions[valid_indices]

    for i, frame in enumerate(dataset):
        assert_array_almost_equal(frame.rotation.as_euler('xyz'),
                                  expected_euler_angles[i])
        assert_array_almost_equal(frame.position, expected_positions[i])
