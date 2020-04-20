import numpy as np
from skimage.color import rgb2gray
from tadataka.dataset import NewTsukubaDataset
from tadataka.vo.semi_dense.frame import ReferenceSelector
from tadataka.vo.semi_dense.frame import Frame
from tests.dataset.path import new_tsukuba


def test_reference_frame_selector():
    def frame(frame_):
        return Frame(frame_.camera_model, rgb2gray(frame_.image), frame_.pose)

    dataset = NewTsukubaDataset(new_tsukuba)
    frame0 = frame(dataset[0][0])
    frame1 = frame(dataset[1][0])
    frame2 = frame(dataset[2][0])
    refframes = [frame0, frame1, frame2]

    M = np.ones((50, 50), dtype=np.int64)
    age_map = np.block([
        [0 * M, 1 * M],
        [2 * M, 3 * M]
    ])
    selector = ReferenceSelector(refframes, age_map)
    #                 x,   y
    assert(selector([25, 25]) is None)
    assert(selector([75, 25]) is frame2)
    assert(selector([25, 75]) is frame1)
    assert(selector([75, 75]) is frame0)
