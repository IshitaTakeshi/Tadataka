import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal, assert_equal)
from scipy.spatial.transform import Rotation

from tadataka.dataset.new_tsukuba import NewTsukubaDataset

from tests.dataset.path import new_tsukuba


def test_new_tsukuba():
    dataset = NewTsukubaDataset(new_tsukuba)

    assert_equal(len(dataset), 5)

    L, R = dataset[0]
    image_shape = (480, 640, 3)
    assert_equal(L.image.shape, image_shape)
    assert_equal(R.image.shape, image_shape)
    assert_equal(L.depth_map.shape, image_shape[0:2])
    assert_equal(R.depth_map.shape, image_shape[0:2])

    L, R = dataset[4]
    translation_true = np.array([-51.802731, 4.731323, -105.171677])
    degrees_true = np.array([16.091024, -10.583960, 0.007110])
    rotation_true = Rotation.from_euler('xyz', degrees_true, degrees=True)

    pose_l = L.pose
    assert_array_almost_equal(
        pose_l.t,
        translation_true + np.dot(rotation_true.as_matrix(), [-5, 0, 0])
    )
    assert_array_almost_equal(
        pose_l.rotation.as_euler('xyz', degrees=True),
        degrees_true
    )

    pose_r = R.pose
    assert_array_almost_equal(
        pose_r.t,
        translation_true + np.dot(rotation_true.as_matrix(), [5, 0, 0])
    )
    assert_array_almost_equal(
        pose_r.rotation.as_euler('xyz', degrees=True),
        degrees_true
    )
