from numpy.testing import assert_array_equal
import numpy as np

from scipy.spatial.transform import Rotation
from tadataka.pose import Pose
from tadataka.camera import CameraModel, CameraParameters
from tadataka.dataset import NewTsukubaDataset
from tests.dataset.path import new_tsukuba
from tadataka.vo.semi_dense.age import AgeMap


def test_increment_age():
    width, height = 800, 1000
    shape = height, width

    camera_model = CameraModel(
        CameraParameters(focal_length=[100, 100],
                         offset=[width / 2, height / 2]),
        distortion_model=None
    )

    # hard to test if rotation is not identity
    pose0 = Pose(Rotation.identity(), np.array([0, 0, 0]))
    pose1 = Pose(Rotation.identity(), np.array([0, 0, -100]))
    pose2 = Pose(Rotation.identity(), np.array([0, 0, -200]))
    depth_map = 100 * np.ones(shape)

    age_map = AgeMap(camera_model, pose0, depth_map)
    age_map0 = age_map.get()
    expected = np.zeros(shape)
    assert_array_equal(age_map0, expected)

    age_map.increment(camera_model, pose1, depth_map)
    age_map1 = age_map.get()

    expected = np.zeros(shape)
    expected[250:750, 200:600] = 1
    assert_array_equal(age_map1, expected)

    age_map.increment(camera_model, pose2, depth_map)
    age_map2 = age_map.get()
    expected = np.zeros(shape)
    expected[250:750, 200:600] = 1
    expected[375:625, 300:500] = 2
    assert_array_equal(age_map2, expected)
