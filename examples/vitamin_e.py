from pathlib import Path

from skimage.io import imread
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import pandas as pd

from skimage import exposure
from tadataka.camera.io import load
from tadataka.feature import extract_features, Matcher
from tadataka.flow_estimation.flow_estimation import estimate_affine_transform
from tadataka.flow_estimation.image_curvature import (
    extract_curvature_extrema, compute_image_curvature)
from tadataka.pose import Pose, estimate_pose_change, solve_pnp
from tadataka.triangulation import Triangulation, TwoViewTriangulation
from tadataka.flow_estimation.extrema_tracker import ExtremaTracker
from tadataka.dataset.tum_rgbd import TumRgbdDataset
from tadataka.utils import is_in_image_range
from tadataka.plot import plot_map, plot_matches


match = Matcher(enable_ransac=False, enable_homography_filter=False)


class DenseKeypointExtractor(object):
    def __init__(self, percentile):
        self.percentile = percentile

    def __call__(self, image):
        return extract_curvature_extrema(image, self.percentile)


extract_dense_keypoints = DenseKeypointExtractor(percentile=80)


def keypoints_from_new_area(image1, flow01):
    """Extract keypoints from newly observed image rea"""
    keypoints1 = extract_dense_keypoints(image1)
    # out of image range after transforming from frame1 to frame0
    # we assume image1.shape == image0.shape
    mask = ~is_in_image_range(flow01.inverse(keypoints1), image1.shape)
    return keypoints1[mask]


def plot_curvature(image, curvature):
    plt.subplot(121)
    plt.imshow(image, cmap="gray")
    plt.subplot(122)
    plt.imshow(curvature, cmap="gray")
    plt.show()


def plot_distance_hist(origin, moved):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("moved distance")
    ax.hist(np.sum(np.power(moved - origin, 2), axis=1), bins=20)
    plt.show()


def track_(keypoints0, image1, flow01):
    image1 = rgb2gray(image1)
    image1 = image1 - np.mean(image1)
    image1 = exposure.equalize_adapthist(image1)
    curvature = compute_image_curvature(image1)

    tracker = ExtremaTracker(curvature, lambda_=1e2)
    keypoints1_ = flow01(keypoints0)
    keypoints1 = tracker.optimize(keypoints1_)
    # plot_distance_hist(keypoints1_, keypoints1)

    return keypoints1


def estimate_flow(features0, features1):
    matches01 = match(features0, features1)
    keypoints0 = features0.keypoints[matches01[:, 0]]
    keypoints1 = features1.keypoints[matches01[:, 1]]
    return estimate_affine_transform(keypoints0, keypoints1)


def plot_track(image1, image2, keypoints1, keypoints2):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(image1)
    ax.scatter(keypoints1[:, 0], keypoints1[:, 1], s=0.1, c='red')
    ax = fig.add_subplot(122)
    ax.imshow(image2)
    ax.scatter(keypoints2[:, 0], keypoints2[:, 1], s=0.1, c='red')
    plt.show()


np.random.seed(3939)


def match_keypoints(keypoint0, keypoint1):
    _, indices0, indices1 = np.intersect1d(
        get_ids(keypoint0), get_ids(keypoint1),
        return_indices=True
    )
    return np.column_stack((indices0, indices1))


def triangulate(pose0, pose1, keypoints0, keypoints1):
    matches01 = match_keypoints(keypoints0, keypoints1)
    triangulator = TwoViewTriangulation(pose0, pose1)
    keypoints0_ = get_array(keypoints0)[matches01[:, 0]]
    keypoints1_ = get_array(keypoints1)[matches01[:, 1]]
    return triangulator.triangulate(
        camera_model.undistort(keypoints0_),
        camera_model.undistort(keypoints1_)
    )


def choose_nonoverlap(size, ratio):
    N = int(size * ratio)
    indices_ = np.arange(0, size)
    np.random.shuffle(indices_)
    indices = indices_[:N]
    return indices


def test_choose_nonoverlap():
    indices = choose_nonoverlap(100, 0.3)
    assert(len(indices) == 30)
    np.testing.assert_array_equal(np.unique(indices), np.sort(indices))


def plot_depth_hist(depths, n_bins=100):
    fig, axs = plt.subplots(1, depths.shape[0], sharey=True, tight_layout=True)
    for i, ax in enumerate(axs):
        ax.hist(depths[i], bins=n_bins)
    plt.show()


test_choose_nonoverlap()


def create_keypoint_frame(start_id, keypoints):
    N = keypoints.shape[0]
    ids = np.arange(start_id, start_id + N)
    return create_keypoint_frame_(ids, keypoints)


def create_first_keypoints(image):
    keypoints = extract_dense_keypoints(image)
    return create_keypoint_frame(0, keypoints)


def get_array(frame):
    return frame[['x', 'y']].to_numpy()


def get_ids(frame):
    return frame['id'].to_numpy()


def create_keypoint_frame_(ids, keypoints):
    assert(keypoints.shape == (ids.shape[0], 2))
    return pd.DataFrame({'id': ids,
                         'x': keypoints[:, 0],
                         'y': keypoints[:, 1]})


class Tracker(object):
    def __init__(self, features0, features1, image1):
        matches01 = match(features0, features1)
        self.flow01 = estimate_affine_transform(
            features0.keypoints[matches01[:, 0]],
            features1.keypoints[matches01[:, 1]]
        )
        self.image1 = image1

    def __call__(self, keypoints0):
        keypoints0_ = get_array(keypoints0)
        keypoints1_ = track_(keypoints0_, self.image1, self.flow01)
        mask1 = is_in_image_range(keypoints1_, self.image1.shape)
        ids0 = get_ids(keypoints0)
        keypoints1 = create_keypoint_frame_(ids0[mask1], keypoints1_[mask1])

        id_start = ids0[-1] + 1
        new_keypoints1 = keypoints_from_new_area(self.image1, self.flow01)
        new_rows = create_keypoint_frame(id_start, new_keypoints1)

        return pd.concat([keypoints1, new_rows])


def test_create_keypoint_frame():
    keypoints = np.arange(10).reshape(5, 2)
    frame = create_keypoint_frame(0, keypoints)
    np.testing.assert_array_equal(frame['id'].to_numpy(), np.arange(5))
    np.testing.assert_array_equal(frame[['x', 'y']].to_numpy(), keypoints)

    frame = create_keypoint_frame(2, keypoints)
    np.testing.assert_array_equal(frame['id'].to_numpy(), np.arange(2, 7))
    np.testing.assert_array_equal(frame[['x', 'y']].to_numpy(), keypoints)


test_create_keypoint_frame()

# camera_model = load("./datasets/saba/cameras.txt")[1]
# filenames = sorted(Path("./datasets/saba/images").glob("*.jpg"))
#
# image0, image1, image2 = [imread(f) for f in filenames[189:192]]

dataset = TumRgbdDataset(Path("datasets", "rgbd_dataset_freiburg1_xyz"))
camera_model = dataset.camera_model
frames = dataset[660:670]
poses = [Pose(f.rotation, f.position).world_to_local() for f in frames]
images = [f.image for f in frames]
features = [extract_features(image) for image in images]

# HACK is it better to drop keypoints0 that are out of the next image range ?

keypoints = [None] * len(frames)

keypoints[0] = create_first_keypoints(images[0])

for i in range(len(keypoints)-1):
    keypoints[i+1] = Tracker(features[i], features[i+1], images[i+1])(keypoints[i])

index0, index1 = 0, -1
# plot_matches(
#     images[index0], images[index1],
#     get_array(keypoints[index0]), get_array(keypoints[index1]),
#     match_keypoints(keypoints[index0], keypoints[index1])
# )

plot_track(images[index0], images[index1],
           get_array(keypoints[index0]), get_array(keypoints[index1]))

points01, depths = triangulate(
    poses[index0], poses[index1],
    keypoints[index0], keypoints[index1]
)
plot_map([poses[index0], poses[index1]], points01)

exit(0)


DO_PNP = True
if DO_PNP:
    poses = [None] * 3
    poses[0] = Pose.identity()

if DO_PNP:
    poses[1] = estimate_pose_change(
        camera_model.undistort(features0.keypoints[matches01[:, 0]]),
        camera_model.undistort(features1.keypoints[matches01[:, 1]])
    )

plot_track(image0, image1,
           tracked_keypoints0[tracked_mask1],
           tracked_keypoints1[tracked_mask1])

triangulator = TwoViewTriangulation(poses[0], poses[1])
points01, depths = triangulator.triangulate(
    camera_model.undistort(tracked_keypoints0[tracked_mask1]),
    camera_model.undistort(tracked_keypoints1[tracked_mask1])
)

plot_map([poses[0], poses[1]], points01)
plot_depth_hist(depths)

tracked_keypoints1 = tracked_keypoints1[tracked_mask1]

points = points01

assert(points01.shape[0] == tracked_keypoints1.shape[0])

flow12 = estimate_flow(features1, features2)

tracked_keypoints2 = track(tracked_keypoints1, image2, flow12)
tracked_mask2 = is_in_image_range(tracked_keypoints2, image2.shape)

if DO_PNP:
    poses[2] = solve_pnp(
        points01[tracked_mask2],
        camera_model.undistort(tracked_keypoints2[tracked_mask2])
    )

tracked_new_keypoints2 = track(new_keypoints1, image2, flow12)
new_mask2 = is_in_image_range(tracked_new_keypoints2, image2.shape)

triangulator = TwoViewTriangulation(poses[1], poses[2])
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
plot_map([poses[0], poses[1], poses[2]], points)
