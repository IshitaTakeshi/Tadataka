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
#                     min_active_keyframes=3)


vo = VisualOdometry(
    CameraParameters(focal_length=[3104.3, 3113.34],
                     offset=[1640, 1232]),
    FOV(-0.01),
    min_active_keyframes=2
)


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


def add_keyframe(image):
    keypoints, descriptors = extract_keypoints(image)
    keypoints_ = vo.camera_model.undistort(keypoints)
    assert(vo.try_add_keyframe(KD(keypoints_, descriptors)))
    return keypoints


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


filenames = sorted(Path("./datasets/ball/").glob("*.jpg"))
images = [rgb2gray(imread(filename)) for filename in filenames[:5]]

keypoints0 = add_keyframe(images[0])
keypoints1 = add_keyframe(images[1])
# plot_matches_(images[0], images[1], keypoints0, keypoints1,
#               point_indices0, point_indices1)
plot_map_(vo.export_poses(), vo.export_points())

keypoints2 = add_keyframe(images[2])
# plot_matches_(images[0], images[2], keypoints0, keypoints2,
#               point_indices0, point_indices2)
# plot_matches_(images[1], images[2], keypoints1, keypoints2,
#               point_indices1, point_indices2)
plot_map_(vo.export_poses(), vo.export_points())

keypoints3 = add_keyframe(images[3])
# plot_matches_(images[0], images[3], keypoints0, keypoints3,
#               point_indices0, point_indices3)
# plot_matches_(images[1], images[3], keypoints1, keypoints3,
#               point_indices1, point_indices3)
# plot_matches_(images[2], images[3], keypoints2, keypoints3,
#               point_indices2, point_indices3)
plot_map_(vo.export_poses(), vo.export_points())

keypoints3 = add_keyframe(images[4])
plot_map_(vo.export_poses(), vo.export_points())
