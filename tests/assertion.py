from numpy.testing import assert_array_almost_equal
from vitamine.rigid_transform import transform


def assert_projection_equal(projection, pose, points, keypoints_pred):
    assert_array_almost_equal(
        projection.compute(transform(pose.R, pose.t, points)),
        keypoints_pred
    )
