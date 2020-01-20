from pathlib import Path

import numpy as np
from skimage.feature import plot_matches
from skimage.io import imread

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from tadataka.coordinates import xy_to_yx
from tadataka.visual_odometry import FeatureBasedVO
from tadataka.camera.io import load
from tadataka.coordinates import local_to_world
from tadataka.plot import plot_map
from tadataka.plot.cameras import cameras_poly3d
from tadataka.pose import Pose


def match_point_indices(point_indices1, point_indices2):
    matches = []
    for i1, p1 in enumerate(point_indices1):
        for i2, p2 in enumerate(point_indices2):
            if p1 == p2:
                matches.append([i1, i2])
    return np.array(matches)


def test_match_point_indices():
    matches = match_point_indices(
        [1, 0, 2, 4, 5, 6],
        [9, 1, 2, 3, 4, 5]
    )

    expected = np.array([
        [0, 1],
        [2, 2],
        [3, 4],
        [4, 5]
    ])
    assert_array_equal(matches, expected)


def plot_matches_(image1, image2, keypoints1, keypoints2,
                  keypoint_point_map1, keypoint_point_map2):
    keypoint_indices1, point_indices1 = zip(*keypoint_point_map1.items())
    keypoint_indices2, point_indices2 = zip(*keypoint_point_map2.items())
    keypoint_indices1 = list(keypoint_indices1)
    keypoint_indices2 = list(keypoint_indices2)

    matches12 = match_point_indices(point_indices1, point_indices2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_matches(ax, image1, image2,
                 xy_to_yx(keypoints1[keypoint_indices1]),
                 xy_to_yx(keypoints2[keypoint_indices2]),
                 matches12)
    plt.show()


def plot_keypoints(image, keypoints):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap="gray")
    ax.scatter(keypoints[:, 0], keypoints[:, 1],
               facecolors='none', edgecolors='r')
    plt.show()


camera_models = load("datasets/saba/cameras.txt")
vo = FeatureBasedVO(camera_models[1], window_size=6)

filenames = sorted(Path("./datasets/saba/images").glob("*.jpg"))
filenames = [filenames[0]] + filenames[4:]
filenames = filenames[:5]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

points = np.empty((0, 3))
points_ax = ax.scatter(points[:, 0], points[:, 1], points[:, 2])
poses = [Pose.identity()]
poses_ax = ax.add_collection3d(cameras_poly3d(poses))


def update_lines(i):
    image = imread(filenames[i])
    viewpoint = vo.add(image)

    if viewpoint < 0:
        return

    vo.try_remove()

    if i == 0:
        return

    poses = vo.export_poses()
    points, colors = vo.export_points()

    # NOTE: there is no .set_data() for 3 dim data...
    points_ax.set_data(points)
    poses_ax.set_data(cameras_poly3d(poses))
    return lines


anim = FuncAnimation(fig, update_lines, len(filenames),
                     interval=1000, blit=False)

plt.show()
