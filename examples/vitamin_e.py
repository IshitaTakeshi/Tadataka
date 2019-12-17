from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

from tadataka.feature import extract_features, Matcher
from tadataka.flow_estimation.flow_estimation import estimate_affine_transform
from tadataka.flow_estimation.image_curvature import (
    extract_curvature_extrema, compute_image_curvature)
from tadataka.pose import Pose, estimate_pose_change, solve_pnp
from tadataka.triangulation import Triangulation, TwoViewTriangulation
from tadataka.flow_estimation.extrema_tracker import ExtremaTracker
from tadataka.dataset.tum_rgbd import TumRgbdDataset
from tadataka.utils import is_in_image_range


def keypoints_from_new_area(image1, flow01):
    """Extract keypoints from newly observed image rea"""
    keypoints1 = extract_curvature_extrema(image1)
    # out of image range after transforming from frame1 to frame0
    # we assume image1.shape == image0.shape
    mask = ~is_in_image_range(flow01.inverse(keypoints1), image1.shape)
    return keypoints1[mask]



def track(keypoints0, image1, flow01):
    curvature = compute_image_curvature(rgb2gray(image1))
    tracker = ExtremaTracker(curvature, lambda_=10.0)
    return tracker.optimize(flow01(keypoints0))


def estimate_flow(features0, features1):
    matches01 = match(features0, features1)
    keypoints0 = features0.keypoints[matches01[:, 0]]
    keypoints1 = features1.keypoints[matches01[:, 1]]
    return estimate_affine_transform(keypoints0, keypoints1)


def plot_track():
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(image1)
    ax.scatter(keypoints[1, :, 0], keypoints[1, :, 1], s=0.1, c='red')
    ax = fig.add_subplot(122)
    ax.imshow(image2)
    ax.scatter(keypoints[2, :, 0], keypoints[2, :, 1], s=0.1, c='red')
    plt.show()


dataset = TumRgbdDataset(Path("datasets", "rgbd_dataset_freiburg1_xyz"))
camera_model = dataset.camera_model
frames = dataset[270:273]


image0 = frames[0].image
image1 = frames[1].image
image2 = frames[2].image

# poses = [Pose(f.rotation, f.position).world_to_local() for f in frames]

match = Matcher(enable_ransac=False, enable_homography_filter=False)

features0 = extract_features(image0)
features1 = extract_features(image1)
features2 = extract_features(image2)

pose0 = Pose.identity()


# HACK is it better to drop keypoints0 that are out of the next image range ?
tracked_keypoints0 = extract_curvature_extrema(image0)

flow01 = estimate_flow(features0, features1)
tracked_keypoints1 = track(tracked_keypoints0, image1, flow01)
tracked_mask1 = is_in_image_range(tracked_keypoints1, image1.shape)
new_keypoints1 = keypoints_from_new_area(image1, flow01)

pose1 = estimate_pose_change(
    camera_model.undistort(tracked_keypoints0[tracked_mask1]),
    camera_model.undistort(tracked_keypoints1[tracked_mask1])
)

triangulator = TwoViewTriangulation(pose0, pose1)
points01, depths = triangulator.triangulate(
    camera_model.undistort(tracked_keypoints0[tracked_mask1]),
    camera_model.undistort(tracked_keypoints1[tracked_mask1])
)

tracked_keypoints1 = tracked_keypoints1[tracked_mask1]

points = points01

assert(points01.shape[0] == tracked_keypoints1.shape[0])

from tadataka.plot import plot_map
plot_map([pose0, pose1], points01)

flow12 = estimate_flow(features1, features2)

tracked_keypoints2 = track(tracked_keypoints1, image2, flow12)
tracked_mask2 = is_in_image_range(tracked_keypoints2, image2.shape)
pose2 = solve_pnp(
    points01[tracked_mask2],
    camera_model.undistort(tracked_keypoints2[tracked_mask2])
)

tracked_new_keypoints2 = track(new_keypoints1, image2, flow12)
new_mask2 = is_in_image_range(tracked_new_keypoints2, image2.shape)

triangulator = TwoViewTriangulation(pose1, pose2)
points12, depths = triangulator.triangulate(
    camera_model.undistort(new_keypoints1[new_mask2]),
    camera_model.undistort(tracked_new_keypoints2[new_mask2])
)

tracked_keypoints2 = np.vstack((
    tracked_keypoints2[tracked_mask2],
    tracked_new_keypoints2[new_mask2]
))
new_keypoints2 = keypoints_from_new_area(image2, flow12)

points = np.vstack((points, points12))
plot_map([pose0, pose1, pose2], points)
