import numpy as np
from skimage.color import rgb2gray, gray2rgb

from tqdm import tqdm
from tadataka.camera import CameraModel
from tadataka.coordinates import image_coordinates
from tadataka.dataset import TumRgbdDataset
from tadataka.vo.semi_dense import SemiDenseVO
from tadataka.vo.semi_dense.frame import Frame
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.matrix import to_homogeneous
from tadataka.rigid_transform import transform
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.vo.semi_dense.propagation import propagate
from tests.dataset.path import new_tsukuba

from examples.plot import plot_depth, plot_prior
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tadataka.vo.semi_dense.semi_dense import InverseDepthMapEstimator


def gray_frame(frame):
    return Frame(frame.camera_model, rgb2gray(frame.image), frame.pose)


def update(keyframe, refframe, prior_inv_depth_map, prior_variance_map):
    estimator = InverseDepthMapEstimator(
        keyframe,
        sigma_i=0.01, sigma_l=0.02,
        step_size_ref=0.01, min_gradient=20.0
    )

    inv_depth_map, variance_map, flag_map = estimator(
        refframe, prior_inv_depth_map, prior_variance_map
    )

    inv_depth_map, variance_map = fusion(prior_inv_depth_map, inv_depth_map,
                                         prior_variance_map, variance_map)

    return inv_depth_map, variance_map, flag_map


dataset = TumRgbdDataset("datasets/rgbd_dataset_freiburg1_xyz",
                         which_freiburg=1)

frames = dataset[490:]

inv_depth_map = np.random.uniform(0.8, 1.2, size=frames[0].depth_map.shape)
# inv_depth_map = invert_depth(frames[0].depth_map)
variance_map = 10.0 * np.ones(frames[0].depth_map.shape)

for i in range(len(frames)):
    kf0_ = frames[i]
    kf1_ = frames[i+1]
    keyframe0 = gray_frame(kf0_)
    keyframe1 = gray_frame(kf1_)
    refframe = gray_frame(frames[i+12])

    plot_prior(keyframe0.image, kf0_.depth_map,
               invert_depth(inv_depth_map), variance_map)

    inv_depth_map, variance_map, flag_map = update(
        keyframe0, refframe, inv_depth_map, variance_map
    )

    plot_depth(keyframe0.image, refframe.image, flag_map,
               kf0_.depth_map, invert_depth(inv_depth_map), variance_map)

    inv_depth_map, variance_map = propagate(keyframe0, keyframe1,
                                            inv_depth_map, variance_map)
