from pathlib import Path
from numpy.testing import assert_equal

from tadataka.dataset.new_tsukuba import NewTsukubaDataset


dataset_root = Path(Path(__file__).parent, "new_tsukuba")


def test_new_tsukuba():
    image_shape = (480, 640, 3)
    dataset = NewTsukubaDataset(dataset_root)

    assert_equal(len(dataset), 5)

    frame = dataset[0]
    assert_equal(frame.image_left.shape, image_shape)
    assert_equal(frame.image_right.shape, image_shape)
    assert_equal(frame.depth_map_left.shape, image_shape[0:2])
    assert_equal(frame.depth_map_right.shape, image_shape[0:2])

    assert(dataset[1].timestamp_rgb == 1.0 / 30)
    assert(dataset[2].timestamp_rgb == 2.0 / 30)

    frames = dataset[1:4:2]

    assert_equal(frames[0].image_left.shape, image_shape)
    assert_equal(frames[0].image_right.shape, image_shape)
    assert_equal(frames[0].depth_map_left.shape, image_shape[0:2])
    assert_equal(frames[0].depth_map_right.shape, image_shape[0:2])

    assert_equal(frames[1].image_left.shape, image_shape)
    assert_equal(frames[1].image_right.shape, image_shape)
    assert_equal(frames[1].depth_map_left.shape, image_shape[0:2])
    assert_equal(frames[1].depth_map_right.shape, image_shape[0:2])

    assert_equal(len(frames), 2)
