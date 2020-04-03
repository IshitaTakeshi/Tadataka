from tadataka.dataset import NewTsukubaDataset
from tadataka.vo.semi_dense.frame_selection import ReferenceFrameSelector
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.vo.semi_dense.frame import Frame
from tests.dataset.path import new_tsukuba


def test_reference_frame_selector():
    dataset = NewTsukubaDataset(new_tsukuba)
    L, R = dataset[0]
    frame0_, frame1_, frame2_ = L, R, L
    frame0 = Frame(frame0_)
    frame1 = Frame(frame1_)
    frame2 = Frame(frame2_)

    selector = ReferenceFrameSelector(frame0, invert_depth(frame0_.depth_map))
    #                  x,   y
    assert(selector([250, 370]) is None)
    assert(selector([100, 200]) is None)

    selector.update(frame1, invert_depth(frame1_.depth_map))
    #                  x,   y
    assert(selector([300, 300]) is frame0)
    assert(selector([610, 300]) is None)

    selector.update(frame2, invert_depth(frame2_.depth_map))
    #                  x,   y
    assert(selector([300, 300]) is frame0)
    assert(selector([360, 250]) is frame1)
    assert(selector([10, 370]) is None)
