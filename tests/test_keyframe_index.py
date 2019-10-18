from numpy.testing import assert_array_equal

from vitamine.keyframe_index import KeyframeIndices


def test_keyframe_indices():
    indices = KeyframeIndices()
    assert(len(indices) == 0)

    indices.add_new(0)
    indices.add_new(1)
    indices.add_new(2)
    assert(len(indices) == 3)

    indices.remove(1)
    assert_array_equal([i for i in indices], [0, 2])

    indices.add_new(3)
    indices.add_new(4)
    assert_array_equal([i for i in indices], [0, 2, 3, 4])
