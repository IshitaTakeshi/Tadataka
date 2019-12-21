from pathlib import Path

from skimage.io import imread
import numpy as np
from matplotlib import pyplot as plt

from tadataka.feature import Matcher
from tadataka.camera.io import load
from tadataka.feature import extract_features
from tadataka.pose import Pose, estimate_pose_change, solve_pnp
from tadataka.triangulation import TwoViewTriangulation
from tadataka.visual_odometry.vitamin_e import (
    Tracker, init_keypoint_frame, get_ids, get_array)
from tadataka.dataset.tum_rgbd import TumRgbdDataset
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
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.scatter(keypoints1[:, 0], keypoints1[:, 1], s=0.1, c='red')
    ax.imshow(image1)

    h, w = image1.shape[0:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    ax = fig.add_subplot(122)
    ax.scatter(keypoints2[:, 0], keypoints2[:, 1], s=0.1, c='red')
    ax.imshow(image2)

    h, w = image2.shape[0:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    plt.show()


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


def match_keypoints(keypoint0, keypoint1):
    _, indices0, indices1 = np.intersect1d(
        get_ids(keypoint0), get_ids(keypoint1),
        return_indices=True
    )
    return np.column_stack((indices0, indices1))


def triangulate(camera_model, pose0, pose1, keypoints0, keypoints1):
    matches01 = match_keypoints(keypoints0, keypoints1)
    keypoints0_ = get_array(keypoints0)[matches01[:, 0]]
    keypoints1_ = get_array(keypoints1)[matches01[:, 1]]
    triangulator = TwoViewTriangulation(pose0, pose1)
    points01, depths = triangulator.triangulate(
        camera_model.undistort(keypoints0_),
        camera_model.undistort(keypoints1_)
    )
    return points01, depths


# camera_model = load("./datasets/saba/cameras.txt")[1]
# filenames = sorted(Path("./datasets/saba/images").glob("*.jpg"))
#
# images = [imread(f) for f in filenames[189:192]]

dataset = TumRgbdDataset(Path("datasets", "rgbd_dataset_freiburg1_desk"))
camera_model = dataset.camera_model
print("len(dataset)", len(dataset))
frames = dataset[520:530]
poses = [Pose(f.rotation, f.position).world_to_local() for f in frames]
images = [f.image for f in frames]

features = [extract_features(image) for image in images]

# HACK is it better to drop keypoints0 that are out of the next image range ?

keypoints = [None] * len(images)
keypoints[0] = init_keypoint_frame(images[0])

for i in range(len(keypoints)-1):
    tracker = Tracker(features[i], features[i+1], images[i+1], lambda_=1e8)
    keypoints[i+1] = tracker(keypoints[i])

index0, index1 = 0, -1
# plot_matches(
#     images[index0], images[index1],
#     get_array(keypoints[index0]), get_array(keypoints[index1]),
#     match_keypoints(keypoints[index0], keypoints[index1])
# )

plot_track(images[index0], images[index1],
           get_array(keypoints[index0]), get_array(keypoints[index1]))

DO_PNP = False

if DO_PNP:
    poses = []
    pose0 = Pose.identity()
    poses.append(pose0)

if DO_PNP:
    match = Matcher(enable_ransac=False, enable_homography_filter=False)
    matches01 = match(features[index0], features[index1])
    pose1 = estimate_pose_change(
        camera_model.undistort(features[index0].keypoints[matches01[:, 0]]),
        camera_model.undistort(features[index1].keypoints[matches01[:, 1]])
    )
    poses.append(pose1)

points01, depths = triangulate(
    camera_model,
    poses[index0], poses[index1],
    keypoints[index0], keypoints[index1]
)
plot_map([poses[index0], poses[index1]], points01)
