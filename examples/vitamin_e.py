from pathlib import Path

from skimage.io import imread
import numpy as np
from matplotlib import pyplot as plt

from tadataka.camera.io import load
from tadataka.feature import extract_features, Matcher
from tadataka.pose import Pose, estimate_pose_change, solve_pnp
from tadataka.triangulation import TwoViewTriangulation, Triangulation
from tadataka.vo.vitamin_e import (
    Tracker, estimate_flow, get_array, init_keypoint_frame,
    match_keypoints, match_multiple_keypoints)
from tadataka.dataset.tum_rgbd import TumRgbdDataset
from tadataka.dataset.euroc import EurocDataset
from tadataka.utils import is_in_image_range
from tadataka.plot import plot_map, plot_matches


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


def plot_track(image1, image2, keypoints1, keypoints2):
    matches12 = match_keypoints(keypoints1, keypoints2)
    indices1, indices2 = matches12[:, 0], matches12[:, 1]

    keypoints1_ = get_array(keypoints1)
    keypoints2_ = get_array(keypoints2)

    colors = np.random.random((matches12.shape[0], 3))

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.scatter(keypoints1_[indices1, 0], keypoints1_[indices1, 1],
               s=0.1, c=colors)
    ax.imshow(image1, cmap="gray")

    h, w = image1.shape[0:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    ax = fig.add_subplot(122)
    ax.scatter(keypoints2_[indices2, 0], keypoints2_[indices2, 1],
               s=0.1, c=colors)
    ax.imshow(image2, cmap="gray")

    h, w = image2.shape[0:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)


np.random.seed(3939)


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


def triangulate(camera_model0, camera_model1,
                pose0, pose1, keypoints0, keypoints1):
    matches01 = match_keypoints(keypoints0, keypoints1)
    keypoints0_ = get_array(keypoints0)[matches01[:, 0]]
    keypoints1_ = get_array(keypoints1)[matches01[:, 1]]
    triangulator = TwoViewTriangulation(pose0, pose1)
    A0 = camera_model0.normalize(keypoints0_)
    A1 = camera_model1.normalize(keypoints1_)

    colors = np.random.random((len(matches01), 3))
    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.scatter(A0[:, 0], A0[:, 1], s=0.1, c=colors)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax = fig.add_subplot(122)
    ax.scatter(A1[:, 0], A1[:, 1], s=0.1, c=colors)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    plt.show()

    points01, depths = triangulator.triangulate(
        camera_model0.normalize(keypoints0_),
        camera_model1.normalize(keypoints1_)
    )
    return points01, depths


def ordinary_triangulation(camera_model0, camera_model1,
                           pose0, pose1, features0, features1):
    match = Matcher()
    matches01 = match(features0, features1)
    triangulator = TwoViewTriangulation(pose0, pose1)
    points01, depths = triangulator.triangulate(
        camera_model0.normalize(features0.keypoints[matches01[:, 0]]),
        camera_model1.normalize(features0.keypoints[matches01[:, 1]])
    )
    plot_map([pose0, pose1], points01)


def match_undistort(camera_models, keypoints_list):
    assert(len(camera_models) == len(keypoints_list))
    # matches.shape == (n_keypoints, n_poses)
    matches = match_multiple_keypoints(keypoints_list)
    print("n matches ", matches.shape[0])

    n_poses = len(keypoints_list)
    n_keypoints = matches.shape[0]

    keypoints = np.empty((n_poses, n_keypoints, 2),
                         dtype=np.float64)

    Z = zip(camera_models, keypoints_list)
    for i, (camera_model, keypoints_) in enumerate(Z):
        A = get_array(keypoints_)
        keypoints[i] = camera_model.undistort(A[matches[:, i]])
    return keypoints


dataset = TumRgbdDataset(Path("datasets/rgbd_dataset_freiburg1_xyz"),
                         which_freiburg=1)
frames = dataset[320:337]
images = [f.image for f in frames]
camera_models = [f.camera_model for f in frames]
poses = [f.pose.world_to_local() for f in frames]

features = [extract_features(image) for image in images]

keypoints = [None] * len(images)
keypoints[0] = init_keypoint_frame(images[0])
for i in range(len(keypoints)-1):
    flow01 = estimate_flow(features[i], features[i+1])
    tracker = Tracker(flow01, images[i+1], lambda_=0.5)
    keypoints[i+1] = tracker(keypoints[i])

# A = match_undistort(camera_models, keypoints)
# points, depths = Triangulation(poses).triangulate(A)
# plot_map(poses, points)
# exit(0)

index0, index1 = 2, -1
plot_track(images[index0], images[index1],
           keypoints[index0], keypoints[index1])

points01, depths = triangulate(
    camera_models[index0], camera_models[index1],
    poses[index0], poses[index1],
    keypoints[index0], keypoints[index1]
)

plot_map([poses[index0], poses[index1]], points01)
