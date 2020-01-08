from pathlib import Path

from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 100

from skimage.io import imread
import numpy as np

from tadataka.visual_odometry.vitamin_e import (
    Tracker, init_keypoint_frame, get_array, match_keypoints)
from tadataka.pose import Pose, estimate_pose_change, solve_pnp
from tadataka.feature import extract_features, Matcher
from tadataka.triangulation import TwoViewTriangulation
from tadataka.camera.io import load
from tadataka.plot import plot_map, plot_matches


def plot_track(image1, image2, keypoints1, keypoints2):
    matches12 = match_keypoints(keypoints1, keypoints2)
    indices1, indices2 = matches12[:, 0], matches12[:, 1]

    keypoints1_ = get_array(keypoints1)
    keypoints2_ = get_array(keypoints2)

    colors = np.random.random((len(matches12), 3))

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.scatter(keypoints1_[indices1, 0], keypoints1_[indices1, 1],
               s=0.1, c=colors)
    ax.imshow(image1)
    ax.axis("off")

    h, w = image1.shape[0:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    ax = fig.add_subplot(122)
    ax.scatter(keypoints2_[indices2, 0], keypoints2_[indices2, 1],
               s=0.1, c=colors)
    ax.imshow(image2)
    ax.axis("off")

    h, w = image2.shape[0:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    plt.show()


def triangulate(camera_model0, camera_model1,
                pose0, pose1, keypoints0, keypoints1):
    matches01 = match_keypoints(keypoints0, keypoints1)
    keypoints0_ = get_array(keypoints0)[matches01[:, 0]]
    keypoints1_ = get_array(keypoints1)[matches01[:, 1]]
    triangulator = TwoViewTriangulation(pose0, pose1)
    points01, depths = triangulator.triangulate(
        camera_model0.undistort(keypoints0_),
        camera_model1.undistort(keypoints1_)
    )
    return points01, depths


def track_dense_keypoints(images, features):
    keypoints = [None] * len(images)
    keypoints[0] = init_keypoint_frame(images[0])
    for i in range(len(keypoints)-1):
        tracker = Tracker(features[i], features[i+1], images[i+1],
                          lambda_=0.001)
        keypoints[i+1] = tracker(keypoints[i])
    return keypoints


def run_vo(camera_model, keypoints0, keypoints1):
    matches01 = match_keypoints(keypoints0, keypoints1)

    keypoints0_ = get_array(keypoints0)
    keypoints1_ = get_array(keypoints1)

    pose0 = Pose.identity()
    pose1 = estimate_pose_change(
        camera_model.undistort(keypoints0_[matches01[:, 0]]),
        camera_model.undistort(keypoints1_[matches01[:, 1]])
    )

    points01, depths = triangulate(
        camera_model, camera_model,
        pose0, pose1,
        keypoints0, keypoints1
    )

    plot_map([pose0, pose1], points01)


root = Path("datasets", "saba_medium")
camera_model = load(Path(root, "cameras.txt"))[1]
filenames = sorted(Path(root, "images").glob("*.jpg"))
images = [imread(f) for f in filenames[180:192]]
features = [extract_features(image) for image in images]

keypoints = track_dense_keypoints(images, features)

index0, index1 = 2, -1
image0, image1 = images[index0], images[index1]
keypoints0, keypoints1 = keypoints[index0], keypoints[index1]
# matches01 = match_keypoints(keypoints0, keypoints1)
# plot_matches(image0, image1,
#              get_array(keypoints0), get_array(keypoints1),
#              matches01)

plot_track(image0, image1, keypoints0, keypoints1)
run_vo(camera_model, keypoints0, keypoints1)
