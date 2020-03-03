from tqdm import tqdm

from tadataka.dataset import NewTsukubaDataset, TumRgbdDataset

from tadataka.vo.semi_dense.__init__ import SemiDenseVO


dataset = NewTsukubaDataset("datasets/NewTsukubaStereoDataset")
# plot_map([keyframe.pose, refframe.pose], np.empty((0, 3)))
vo = SemiDenseVO()
frame_l, frame_r = dataset[0]
vo.estimate(frame_l)
vo.estimate(frame_r)

# for frame_l, frame_r in dataset[:2]:
#     pose = vo.estimate(frame_l)
