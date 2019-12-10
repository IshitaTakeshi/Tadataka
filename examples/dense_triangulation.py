import numpy as np

from skimage.color import rgb2gray
from skimage.feature import BRIEF

from matplotlib import pyplot as plt

from tadataka.camera import CameraModel, CameraParameters, FOV
from tadataka.coordinates import yx_to_xy, xy_to_yx
from tadataka.dataset.new_tsukuba import NewTsukubaDataset
from tadataka.feature import extract_features, Features, Matcher
from tadataka.flow_estimation.extrema_tracker import ExtremaTracker
from tadataka.flow_estimation.image_curvature import extract_curvature_extrema
from tadataka.flow_estimation.image_curvature import compute_image_curvature
from tadataka.flow_estimation.flow_estimation import estimate_affine_transform
from tadataka.plot import plot_map, plot_matches
from tadataka.pose import Pose
from tadataka.triangulation import Triangulation
from tadataka.utils import is_in_image_range
from matplotlib import rcParams
# rcParams["savefig.dpi"] = 800


brief = BRIEF(
    # descriptor_size=512,
    # patch_size=64,
    # mode="uniform",
    # sigma=0.1
)


camera_model = CameraModel(
    CameraParameters(focal_length=[615, 615], offset=[320, 240]),
    FOV(0.0)
)

match = Matcher()


def extract_dense_features(image):
    image = rgb2gray(image)
    keypoints = extract_curvature_extrema(image, percentile=95)
    keypoints = xy_to_yx(keypoints)
    brief.extract(image, keypoints)
    keypoints = keypoints[brief.mask]
    keypoints = yx_to_xy(keypoints)

    return Features(keypoints, brief.descriptors)


def sparse_triangulation(image0, image1, pose0, pose1):
    features0 = extract_features(image0)
    features1 = extract_features(image1)
    matches01 = match(features0, features1)

    plot_matches(image0, image1,
                 features0.keypoints, features1.keypoints,
                 matches01)

    undistorted_keypoints0 = camera_model.undistort(features0.keypoints)
    undistorted_keypoints1 = camera_model.undistort(features1.keypoints)
    points, depth_mask = Triangulation(pose0, pose1).triangulate(
        undistorted_keypoints0[matches01[:, 0]],
        undistorted_keypoints1[matches01[:, 1]]
    )

    plot_map([pose0, pose1], points)


def dense_triangulation(image0, image1, pose0, pose1):
    features0 = extract_dense_features(image0)
    features1 = extract_dense_features(image1)
    matches01 = match(features0, features1)

    plot_matches(image0, image1,
                 features0.keypoints, features1.keypoints,
                 matches01)

    undistorted_keypoints0 = camera_model.undistort(features0.keypoints)
    undistorted_keypoints1 = camera_model.undistort(features1.keypoints)
    points, depth_mask = Triangulation(pose0, pose1).triangulate(
        undistorted_keypoints0[matches01[:, 0]],
        undistorted_keypoints1[matches01[:, 1]]
    )

    plot_map([pose0, pose1], points)


def vitamine(image0, image1, pose0, pose1):
    features0 = extract_dense_features(image0)
    features1 = extract_dense_features(image1)
    matches01 = match(features0, features1)

    affine = estimate_affine_transform(
        features0.keypoints[matches01[:, 0]],
        features1.keypoints[matches01[:, 1]]
    )

    dense_keypoints0 = extract_curvature_extrema(image0)
    dense_keypoints1 = affine.transform(dense_keypoints0)

    mask = is_in_image_range(dense_keypoints1, image1.shape)

    et = ExtremaTracker(compute_image_curvature(rgb2gray(image1)), lambda_=0.0)
    dense_keypoints1[mask] = et.optimize(dense_keypoints1[mask])

    plot_matches(image0, image1, dense_keypoints0, dense_keypoints1,
                 np.empty((0, 2), dtype=np.int64),
                 keypoints_color='red')

    points, depth_mask = Triangulation(pose0, pose1).triangulate(
        camera_model.undistort(dense_keypoints0),
        camera_model.undistort(dense_keypoints1)
    )

    plot_map([pose0, pose1], points)


dataset_root = "NewTsukubaStereoDataset"

dataset = NewTsukubaDataset(dataset_root)

frame0 = dataset[208]
frame1 = dataset[209]

image0 = frame0.image_left
pose0 = Pose(frame0.rotation, frame0.position_left).world_to_local()

image1 = frame1.image_right
pose1 = Pose(frame1.rotation, frame1.position_right).world_to_local()

sparse_triangulation(image0, image1, pose0, pose1)
dense_triangulation(image0, image1, pose0, pose1)
vitamine(image0, image1, pose0, pose1)
