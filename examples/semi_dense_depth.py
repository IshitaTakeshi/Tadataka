import numpy as np
from skimage.color import rgb2gray, gray2rgb

from tqdm import tqdm
from tadataka.coordinates import image_coordinates
from tadataka.dataset import TumRgbdDataset
from tadataka.vo.semi_dense import SemiDenseVO
from tadataka.vo.semi_dense.frame import Frame
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.matrix import to_homogeneous
from tadataka.rigid_transform import transform
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tests.dataset.path import new_tsukuba

from examples.plot import plot_depth
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


dataset = TumRgbdDataset(
    "datasets/rgbd_dataset_freiburg1_xyz",
    which_freiburg=1
)

from tadataka.vo.semi_dense.semi_dense import InverseDepthEstimator

def estimate(estimator, refframe,
             prior_depth_map, prior_variance_map):
    image_shape = prior_depth_map.shape

    depth_map = np.zeros(image_shape)
    variance_map = np.zeros(image_shape)
    flag_map = np.zeros(image_shape)
    for u_key in tqdm(image_coordinates(image_shape)):
        x, y = u_key
        inv_depth, variance, flag = estimator(
            refframe, u_key,
            prior_depth_map[y, x],
            prior_variance_map[y, x]
        )
        depth_map[y, x] = invert_depth(inv_depth)
        variance_map[y, x] = variance
        flag_map[y, x] = flag

    return depth_map, variance_map, flag_map

kf = dataset[187]
rf = dataset[192]

keyframe = Frame(kf.camera_model, rgb2gray(kf.image), kf.pose)
refframe = Frame(rf.camera_model, rgb2gray(rf.image), rf.pose)

estimator = InverseDepthEstimator(
    keyframe,
    sigma_l=0.005, sigma_i=0.04,
    step_size_ref=0.005, min_gradient=10.0
)

prior_depth_map = np.random.uniform(0.8, 1.2, size=kf.depth_map.shape)
prior_variance_map = 1.0 * np.ones(kf.depth_map.shape)

depth_map, variance_map, flag_map = estimate(
    estimator, refframe,
    prior_depth_map, prior_variance_map
)

plot_depth(kf.image, rf.image,
           flag_map, kf.depth_map,
           depth_map, variance_map)
