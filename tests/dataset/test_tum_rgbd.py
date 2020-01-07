from pathlib import Path

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
from scipy.spatial.transform import Rotation

from tadataka.dataset.tum_rgbd import TumRgbdDataset


dataset_root = Path(Path(__file__).parent, "tum_rgbd")


angles_gt = np.repeat(np.arange(0., 0.7, 0.02), 3).reshape(35, 3)
positions_gt = np.arange(0., 0.21, 0.002).reshape(35, 3)

def test_tum_dataset():
    # 3rd frame should not be loaded because
    # the depth timestamp cannot match the corresponding pose timestamp

    dataset = TumRgbdDataset(dataset_root)
    image_shape = (30, 40)

    # test index access
    assert_equal(len(dataset), 7)
    assert_equal(len(dataset[1:4:2]), 2)

    frame = dataset[0]
    assert_equal(frame.image.shape[0:2], image_shape)
    assert_equal(frame.depth_map.shape[0:2], image_shape)
    assert_equal(frame.image.shape[2], 3)

    indices = [0, 6, 10, 15, 20, 25, 30]
    angles_expected = angles_gt[indices]
    positions_expected = positions_gt[indices]
    for i, frame in enumerate(dataset):
        pose = frame.pose
        assert_array_almost_equal(pose.rotation.as_euler('xyz'),
                                  angles_expected[i])
        assert_array_almost_equal(pose.t, positions_expected[i])
