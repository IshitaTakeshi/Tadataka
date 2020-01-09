from pathlib import Path

from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 100

from skimage.io import imread
import numpy as np

from tadataka.camera.io import load
from tadataka.dataset.tum_rgbd import TumRgbdDataset
from tadataka.feature import extract_features, Matcher
from tadataka.plot import plot_map, plot_matches
from tadataka.pose import Pose, estimate_pose_change, solve_pnp
from tadataka.triangulation import TwoViewTriangulation
from tadataka.visual_odometry.vitamin_e import (
    Tracker, init_keypoint_frame, get_array, match_keypoints)


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
    ax.imshow(image1, cmap="gray")
    ax.axis("off")

    h, w = image1.shape[0:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    ax = fig.add_subplot(122)
    ax.scatter(keypoints2_[indices2, 0], keypoints2_[indices2, 1],
               s=0.1, c=colors)
    ax.imshow(image2, cmap="gray")
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


def run_vo(camera_model0, camera_model1, keypoints0, keypoints1):
    matches01 = match_keypoints(keypoints0, keypoints1)

    keypoints0_ = get_array(keypoints0)
    keypoints1_ = get_array(keypoints1)

    pose0 = Pose.identity()
    pose1 = estimate_pose_change(
        camera_model0.undistort(keypoints0_[matches01[:, 0]]),
        camera_model1.undistort(keypoints1_[matches01[:, 1]])
    )

    points01, depths = triangulate(
        camera_model0, camera_model1,
        pose0, pose1,
        keypoints0, keypoints1
    )

    plot_map([pose0, pose1], points01)


# root = Path("datasets", "saba_medium")
# camera_model = load(Path(root, "cameras.txt"))[1]
# filenames = sorted(Path(root, "images").glob("*.jpg"))
# images = [imread(f) for f in filenames[180:192]]
# features = [extract_features(image) for image in images]


from tadataka.camera import RadTan, CameraParameters, CameraModel

camera_model = CameraModel(
    CameraParameters(focal_length=[172.98992851, 172.98303181],
                     offset=[163.33639726, 134.99537889]),
    RadTan([-0.02757673, -0.00659358, 0.00085669, -0.000309])
)

root = Path("datasets", "indoor_forward_12_davis")
filenames = [Path(root, "img", "image_0_{}.png".format(i)) for i in range(0, 1221)]
images = [imread(f) for f in filenames[780:800]]
camera_models = [camera_model] * len(images)

features = [extract_features(image) for image in images]

keypoints = track_dense_keypoints(images, features)

index0, index1 = 2, -1
image0, image1 = images[index0], images[index1]
keypoints0, keypoints1 = keypoints[index0], keypoints[index1]
camera_model0, camera_model1 = camera_models[index0], camera_models[index1]

matches01 = match_keypoints(keypoints0, keypoints1)
plot_track(image0, image1, keypoints0, keypoints1)
run_vo(camera_model0, camera_model1, keypoints0, keypoints1)
