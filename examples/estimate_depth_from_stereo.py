import numpy as np
from tqdm import tqdm
from skimage.color import rgb2gray
from tadataka.dataset import NewTsukubaDataset
from tadataka.coordinates import image_coordinates
from tadataka.vo.semi_dense.hypothesis import Hypothesis
from tadataka.vo.semi_dense.semi_dense import InvDepthEstimator
from tadataka.numeric import safe_invert
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from examples.plot import plot_depth


dataset = NewTsukubaDataset("datasets/NewTsukubaStereoDataset",
                            condition="fluorescent")
min_depth, prior_depth, max_depth = 60.0, 200.0, 1000.0
prior_variance = 1.0
min_inv_depth = safe_invert(max_depth)
prior_inv_depth = safe_invert(prior_depth)
max_inv_depth = safe_invert(min_depth)

keyframe_, refframe_ = dataset[0]
image_key = rgb2gray(keyframe_.image)
estimate = InvDepthEstimator(keyframe_.camera_model, image_key,
                             [min_inv_depth, max_inv_depth],
                             sigma_i=0.1, sigma_l=0.2,
                             step_size_ref=0.01, min_gradient=0.2)
pose_wk = keyframe_.pose
pose_wr = refframe_.pose
pose_rk = pose_wr.inv() * pose_wk
T_rk = pose_rk.T
image_ref = rgb2gray(refframe_.image)
prior = Hypothesis(prior_inv_depth, prior_variance)

inv_depth_map = prior.inv_depth * np.ones(image_key.shape)
variance_map = prior.variance * np.ones(image_key.shape)
flag_map = np.full(image_key.shape, FLAG.NOT_PROCESSED)
for u_key in tqdm(image_coordinates(image_key.shape)):
    (inv_depth, variance), flag = estimate(refframe_.camera_model, image_ref,
                                           T_rk, u_key, prior)
    x, y = u_key
    inv_depth_map[y, x] = inv_depth
    variance_map[y, x] = variance
    flag_map[y, x] = flag


# plot_depth(image_key, np.zeros(image_key.shape),
#            flag_map, keyframe_.depth_map,
#            safe_invert(inv_depth_map), variance_map)
