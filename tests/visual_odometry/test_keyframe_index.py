from numpy.testing import assert_array_equal

from vitamine.visual_odometry.keyframe_index import KeyframeIndices


def test_keyframe_indices():
    indices = KeyframeIndices()
    assert(len(indices) == 0)

    indices.add_new()
    indices.add_new()
    indices.add_new()
    assert(len(indices) == 3)

    indices.remove(1)
    assert_array_equal([i for i in indices], [0, 2])

    indices.add_new()
    indices.add_new()
    assert_array_equal([i for i in indices], [0, 2, 3, 4])
