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
from tadataka.vo.semi_dense.fusion import fusion
from tests.dataset.path import new_tsukuba

from examples.plot import plot_depth
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


dataset = TumRgbdDataset(
    "datasets/rgbd_dataset_freiburg1_desk",
    which_freiburg=1
)

from tadataka.vo.semi_dense.semi_dense import InverseDepthMapEstimator

kf = dataset[0]
rf = dataset[5]

keyframe = Frame(kf.camera_model, rgb2gray(kf.image), kf.pose)
refframe = Frame(rf.camera_model, rgb2gray(rf.image), rf.pose)

estimator = InverseDepthMapEstimator(
    keyframe,
    sigma_i=0.01, sigma_l=0.02,
    step_size_ref=0.005, min_gradient=10.0
)
image_shape = keyframe.image.shape

prior_inv_depth_map = np.random.uniform(0.8, 1.2, size=kf.depth_map.shape)
prior_variance_map = 1.0 * np.ones(image_shape)

inv_depth_map, variance_map, flag_map = estimator(
    refframe, prior_inv_depth_map, prior_variance_map
)


plot_depth(kf.image, rf.image, flag_map,
           kf.depth_map, invert_depth(inv_depth_map), variance_map)
