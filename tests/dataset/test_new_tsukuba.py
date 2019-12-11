from pathlib import Path
from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal, assert_equal)

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

    assert_array_equal(frame.position_left, [-5, 0, 0])
    assert_array_equal(frame.position_right, [5, 0, 0])

    frame = dataset[4]
    assert_array_almost_equal(
        frame.position_left,
        [-4.99999376e+00, -6.10864598e-07, -3.51827802e-02]
    )
    assert_array_almost_equal(
        frame.position_right,
        [4.99999576e+00,  6.10864598e-07, -4.96552198e-02]
    )

    assert_array_almost_equal(
        frame.rotation.as_euler('xyz', degrees=True),
        [-0.070745, 0.082921, 0.000007]
    )
