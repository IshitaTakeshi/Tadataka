from autograd import numpy as np
from skimage.feature import plot_matches
from skimage.io import imread
from skimage.color import rgb2gray
from pathlib import Path
from vitamine.keypoints import extract_keypoints
from matplotlib import pyplot as plt
from vitamine.coordinates import xy_to_yx
from vitamine.dataset.tum_rgbd import TUMDataset
from vitamine.visual_odometry.visual_odometry import VisualOdometry
from vitamine.visual_odometry.keypoint import LocalFeatures
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
    min_active_keyframes=3
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
    print(image.shape)
    keypoints, descriptors = extract_keypoints(image)
    keypoints_ = vo.camera_model.undistort(keypoints)
    lf = LocalFeatures(keypoints_, descriptors)
    vo.try_add_keyframe(lf)
    # keypoints has the same distortion as image
    return lf, keypoints


def plot_matches_(image1, image2, lf1, lf2, keypoints1, keypoints2):
    matches12 = match_point_indices(
        lf1.point_indices[lf1.is_triangulated],
        lf2.point_indices[lf2.is_triangulated]
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_matches(ax, image1, image2,
                 xy_to_yx(keypoints1[lf1.is_triangulated]),
                 xy_to_yx(keypoints2[lf2.is_triangulated]),
                 matches12[[100, 220, 257, 270]],
                 matches_color='yellow')
    plt.show()


def plot_keypoints(image, keypoints):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap="gray")
    ax.scatter(keypoints[:, 0], keypoints[:, 1], facecolors='none', edgecolors='r')
    plt.show()


def plot_map_(poses, points):
    omegas, translations = zip(*poses)
    omegas = np.array(omegas)
    translations = np.array(translations)
    plot_map(*camera_to_world(omegas, translations), points)


filenames = sorted(Path("./datasets/ball/").glob("*.jpg"))
images = [rgb2gray(imread(filename)) for filename in filenames[:4]]

lf0, keypoints0 = add_keyframe(images[0])
lf1, keypoints1 = add_keyframe(images[1])

plot_map_(vo.export_poses(), vo.export_points())

print(f"len(keypoints0) = {len(keypoints0)}")
print(f"len(keypoints1) = {len(keypoints1)}")
# plot_keypoints(images[0], keypoints0)
plot_matches_(images[0], images[1], lf0, lf1, keypoints0, keypoints1)

lf2, keypoints2 = add_keyframe(images[2])
print(f"len(keypoints2) = {len(keypoints2)}")
plot_matches_(images[0], images[2], lf0, lf2, keypoints0, keypoints2)
plot_matches_(images[1], images[2], lf1, lf2, keypoints1, keypoints2)

lf3, keypoints3 = add_keyframe(images[3])
print(f"len(keypoints3) = {len(keypoints3)}")
plot_matches_(images[0], images[3], lf0, lf3, keypoints0, keypoints3)
plot_matches_(images[1], images[3], lf1, lf3, keypoints1, keypoints3)
plot_matches_(images[2], images[3], lf2, lf3, keypoints2, keypoints3)
