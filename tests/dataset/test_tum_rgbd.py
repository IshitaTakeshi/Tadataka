from pathlib import Path

from numpy.testing import assert_equal
import os

from vitamine.dataset.tum_rgbd import TUMDataset


dataset_root = Path(__file__).parent


def test_tum_dataset():
    dataset = TUMDataset(dataset_root)

    assert_equal(len(dataset), 5)

    frame = dataset[0]
    assert_equal(frame.timestamp_rgb, 0.1)
    assert_equal(frame.timestamp_depth, 0.2)
    assert_equal(frame.image.shape[0:2],
                 frame.depth_map.shape[0:2])
    assert_equal(frame.image.shape[2], 3)

    frames = dataset[1:4:2]

    assert_equal(frames[0].timestamp_rgb, 1.1)
    assert_equal(frames[0].timestamp_depth, 1.2)
    assert_equal(frames[0].image.shape[0:2],
                 frames[0].depth_map.shape[0:2])
    assert_equal(frames[0].image.shape[2], 3)

    assert_equal(frames[1].timestamp_rgb, 3.1)
    assert_equal(frames[1].timestamp_depth, 3.2)
    assert_equal(frames[1].image.shape[0:2],
                 frames[1].depth_map.shape[0:2])
    assert_equal(frames[1].image.shape[2], 3)

    assert_equal(len(frames), 2)
