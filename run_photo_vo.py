from autograd import numpy as np
from skimage.feature import plot_matches
from skimage.io import imread
from skimage.color import rgb2gray
from pathlib import Path
from vitamine.keypoints import extract_keypoints
from vitamine.keypoints import KeypointDescriptor as KD
from matplotlib import pyplot as plt
from vitamine.coordinates import xy_to_yx
from vitamine.dataset.tum_rgbd import TUMDataset
from vitamine.visual_odometry.visual_odometry import VisualOdometry
from vitamine.camera import CameraParameters
from vitamine.camera_distortion import FOV
from vitamine.coordinates import camera_to_world
from vitamine.plot.map import plot_map

# camera_parameters = CameraParameters(
#     focal_length=[525.0, 525.0],
#     offset=[319.5, 239.5]
# )
# dataset = TUMDataset(Path("datasets", "TUM", "rgbd_dataset_freiburg1_xyz"))
# vo = VisualOdometry(camera_parameters, FOV(0.0),
#                     max_active_keyframes=3)

# saba
vo = VisualOdometry(
    CameraParameters(focal_length=[2890.16, 3326.04], offset=[1640, 1232]),
    FOV(0.01),
    max_active_keyframes=4
)

# ball
# vo = VisualOdometry(
#     CameraParameters(focal_length=[3104.3, 3113.34], offset=[1640, 1232]),
#     FOV(-0.01),
#     max_active_keyframes=3
# )


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


def plot_map_(poses, points):
    omegas, translations = zip(*poses)
    omegas = np.array(omegas)
    translations = np.array(translations)
    plot_map(*camera_to_world(omegas, translations), points)


filenames = sorted(Path("./datasets/saba/").glob("*.jpg"))
filenames = [filenames[0]] + filenames[4:]
# import cv2
# images = [cv2.imread(str(filename)) for filename in filenames[0:14]]


for i, filename in enumerate(filenames):
    image = rgb2gray(imread(filename))
    print("Adding {}-th frame".format(i))
    print("filename = {}".format(filename))

    viewpoint = vo.add(image)

    if viewpoint < 0:
        continue

    vo.try_remove()
    print("{}-th Frame Added".format(i))

    if i == 0:
        continue
    plot_map_(vo.export_poses(), vo.export_points())
