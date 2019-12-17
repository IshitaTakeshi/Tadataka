from pathlib import Path

import numpy as np
from skimage.color import rgb2gray
from tadataka.feature import extract_features, Matcher
from tadataka.flow_estimation.flow_estimation import estimate_affine_transform
from tadataka.flow_estimation.image_curvature import (
    extract_curvature_extrema, compute_image_curvature)
from tadataka.pose import Pose, estimate_pose_change, solve_pnp
from tadataka.triangulation import TwoViewTriangulation
from tadataka.flow_estimation.extrema_tracker import ExtremaTracker
from tadataka.dataset.tum_rgbd import TumRgbdDataset
from tadataka.utils import is_in_image_range


def grow(keypoints, keypoints_tracked, keypoints_new):
    nans = np.full((keypoints.shape[0], keypoints_new.shape[0], 2), np.nan)
    # [existing keypoints             nan]
    # [ tracked keypoints   new keypoints]
    upper = np.concatenate((keypoints, nans), axis=1)
    bottom = np.concatenate((keypoints_tracked, keypoints_new), axis=0)
    lower = bottom.reshape(1, *bottom.shape)
    return np.concatenate((upper, lower))


def allocate(points, n_keypoints):
    nans = np.full((n_keypoints, 3), np.nan)
    return np.vstack((points, nans))


def keypoints_from_new_area(image1, flow01):
    keypoints1 = extract_curvature_extrema(image1)
    # out of image range after transforming from frame1 to frame0
    # we assume image1.shape == image0.shape
    mask = ~is_in_image_range(flow01.inverse(keypoints1), image1.shape)
    return keypoints1[mask]


def track(keypoints0, flow01, image1):
    curvature = compute_image_curvature(rgb2gray(image1))
    tracker = ExtremaTracker(curvature, lambda_=10.0)
    keypoints1 = flow01(keypoints0)
    mask = is_in_image_range(keypoints1, curvature.shape)
    keypoints1[mask] = tracker.optimize(keypoints1[mask])
    return keypoints1


dataset = TumRgbdDataset(Path("datasets", "rgbd_dataset_freiburg1_xyz"))
camera_model = dataset.camera_model
frames = dataset[270:274]


image0 = frames[0].image
image1 = frames[1].image
image2 = frames[2].image

# poses = [Pose(f.rotation, f.position).world_to_local() for f in frames]

match = Matcher(enable_ransac=False, enable_homography_filter=False)

features0 = extract_features(image0)
features1 = extract_features(image1)
features2 = extract_features(image2)

matches01 = match(features0, features1)
flow01 = estimate_affine_transform(
    features0.keypoints[matches01[:, 0]],
    features1.keypoints[matches01[:, 1]]
)

keypoints0 = extract_curvature_extrema(image0)
keypoints = keypoints0.reshape(1, *keypoints0.shape)
points = np.full((keypoints.shape[1], 3), np.nan)

keypoints1_tracked = track(keypoints[-1], flow01, image1)
mask = is_in_image_range(keypoints1_tracked, image1.shape)
keypoints1_tracked[~mask] = np.nan
keypoints1_new = keypoints_from_new_area(image1, flow01)
keypoints = grow(keypoints, keypoints1_tracked, keypoints1_new)
points = allocate(points, keypoints1_new.shape[0])
assert(points.shape[0] == keypoints.shape[1])

mask = np.all(~np.isnan(keypoints), axis=(0, 2))

poses = [None] * 3
poses[0] = Pose.identity()
poses[1] = estimate_pose_change(
    camera_model.undistort(keypoints[0, mask]),
    camera_model.undistort(keypoints[1, mask])
)

# poses[0], poses[1] = poses[0], poses[1]
triangulator = TwoViewTriangulation(poses[0], poses[1])
points[mask], depth_mask = triangulator.triangulate(
    camera_model.undistort(keypoints[0, mask]),
    camera_model.undistort(keypoints[1, mask])
)

from tadataka.plot import plot3d
point_mask = np.all(~np.isnan(points), axis=1)
plot3d(points[point_mask])

matches12 = match(features1, features2)

flow12 = estimate_affine_transform(
    features1.keypoints[matches12[:, 0]],
    features2.keypoints[matches12[:, 1]]
)

keypoints2_tracked = track(keypoints[-1], flow12, image2)
mask = is_in_image_range(keypoints2_tracked, image2.shape)
keypoints2_tracked[~mask] = np.nan
keypoints2_new = keypoints_from_new_area(image2, flow12)
keypoints = grow(keypoints, keypoints2_tracked, keypoints2_new)
points = allocate(points, keypoints2_new.shape[0])
assert(points.shape[0] == keypoints.shape[1])

keypoints_ = keypoints[-1]

point_mask = np.all(~np.isnan(points), axis=1)
keypoint_mask = np.all(~np.isnan(keypoints_), axis=1)
mask = point_mask & keypoint_mask
poses[2] = solve_pnp(points[mask], camera_model.undistort(keypoints_[mask]))

from tadataka.plot import plot_map
plot_map(poses, points[point_mask])

keypoint_mask = np.all(~np.isnan(keypoints[-2:]), axis=(0, 2))
point_mask = np.all(~np.isnan(points), axis=1)

mask = keypoint_mask & point_mask
triangulator = TwoViewTriangulation(poses[0], poses[2])
points[mask], depth_mask = triangulator.triangulate(
    camera_model.undistort(keypoints[0, mask]),
    camera_model.undistort(keypoints[2, mask])
)

point_mask = np.all(~np.isnan(points), axis=1)
plot_map(poses, points[point_mask])
