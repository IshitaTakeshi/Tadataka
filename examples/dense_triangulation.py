from pathlib import Path

import numpy as np

from skimage.color import rgb2gray
from skimage.feature import BRIEF

from matplotlib import pyplot as plt

from tadataka.camera import CameraModel, CameraParameters, FOV
from tadataka.coordinates import yx_to_xy, xy_to_yx
from tadataka.dataset.new_tsukuba import NewTsukubaDataset
from tadataka.feature import extract_features, Features, Matcher
from tadataka.triangulation import Triangulation
from tadataka.flow_estimation.extrema_tracker import ExtremaTracker
from tadataka.flow_estimation.image_curvature import extract_curvature_extrema
from tadataka.flow_estimation.image_curvature import compute_image_curvature
from tadataka.flow_estimation.flow_estimation import estimate_affine_transform
from tadataka.plot import plot_map, plot_matches
from tadataka.pose import Pose
from tadataka.triangulation import TwoViewTriangulation
from tadataka.utils import is_in_image_range
from matplotlib import rcParams
# rcParams["savefig.dpi"] = 800


brief = BRIEF(
    # descriptor_size=512,
    # patch_size=64,
    # mode="uniform",
    # sigma=0.1
)


match = Matcher()


def extract_dense_features(image):
    image = rgb2gray(image)
    keypoints = extract_curvature_extrema(image, percentile=95)
    keypoints = xy_to_yx(keypoints)
    brief.extract(image, keypoints)
    keypoints = keypoints[brief.mask]
    keypoints = yx_to_xy(keypoints)

    return Features(keypoints, brief.descriptors)


def sparse_triangulation(frame0, frame1):
    pose0 = Pose(frame0.rotation, frame0.position).world_to_local()
    pose1 = Pose(frame1.rotation, frame1.position).world_to_local()

    features0 = extract_features(frame0.image)
    features1 = extract_features(frame1.image)
    matches01 = match(features0, features1)

    plot_matches(image0, image1,
                 features0.keypoints, features1.keypoints,
                 matches01)

    undistorted_keypoints0 = frame0.camera_model.undistort(features0.keypoints)
    undistorted_keypoints1 = frame1.camera_model.undistort(features1.keypoints)
    points, depth_mask = TwoViewTriangulation(pose0, pose1).triangulate(
        undistorted_keypoints0[matches01[:, 0]],
        undistorted_keypoints1[matches01[:, 1]]
    )

    plot_map([pose0, pose1], points)


def dense_match_triangulation(frame0, frame1):
    features0 = extract_dense_features(frame0.image)
    features1 = extract_dense_features(frame1.image)
    matches01 = match(features0, features1)

    plot_matches(image0, image1,
                 features0.keypoints, features1.keypoints,
                 matches01)

    undistorted_keypoints0 = frame0.camera_model.undistort(features0.keypoints)
    undistorted_keypoints1 = frame1.camera_model.undistort(features1.keypoints)

    points, depth_mask = TwoViewTriangulation(pose0, pose1).triangulate(
        undistorted_keypoints0[matches01[:, 0]],
        undistorted_keypoints1[matches01[:, 1]]
    )

    plot_map([pose0, pose1], points)



def dense_track_triangulation(frame0, frame1):
    features0 = extract_dense_features(image0)
    features1 = extract_dense_features(image1)
    matches01 = match(features0, features1)

    affine = estimate_affine_transform(
        features0.keypoints[matches01[:, 0]],
        features1.keypoints[matches01[:, 1]]
    )

    dense_keypoints0 = extract_curvature_extrema(image0)
    dense_keypoints1 = affine.transform(dense_keypoints0)

    mask = is_in_image_range(dense_keypoints1, image1.shape)

    et = ExtremaTracker(compute_image_curvature(rgb2gray(image1)),
                        lambda_=10.0)
    dense_keypoints1[mask] = et.optimize(dense_keypoints1[mask])

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(image0)
    ax.scatter(dense_keypoints0[mask, 0],
               dense_keypoints0[mask, 1],
               s=0.1, c='red')
    ax = fig.add_subplot(122)
    ax.imshow(image1)
    ax.scatter(dense_keypoints1[mask, 0],
               dense_keypoints1[mask, 1],
               s=0.1, c='red')
    plt.show()

    points, depth_mask = TwoViewTriangulation(pose0, pose1).triangulate(
        frame0.camera_model.undistort(dense_keypoints0[mask]),
        frame1.camera_model.undistort(dense_keypoints1[mask])
    )

    plot_map([pose0, pose1], points)


class Tracker(object):
    def __init__(self, keypoints0, keypoints1, image1):
        assert(keypoints0.shape == keypoints1.shape)
        self.affine = estimate_affine_transform(keypoints0, keypoints1)
        self._tracker = ExtremaTracker(
            compute_image_curvature(rgb2gray(image1)),
            lambda_=10.0
        )
        self.image_shape = image1.shape

    def __call__(self, dense_keypoints0):
        dense_keypoints1 = self.affine(dense_keypoints0)

        mask = is_in_image_range(dense_keypoints1, self.image_shape)

        dense_keypoints1[mask] = self._tracker.optimize(dense_keypoints1[mask])
        return dense_keypoints1


def undistort(camera_models, dense_keypoints):
    assert(len(camera_models) == dense_keypoints.shape[1])
    for i in range(len(camera_models)):
        dense_keypoints[:, i] = camera_models[i].undistort(
            dense_keypoints[:, i]
        )
    return dense_keypoints


def track(trackers, dense_keypoints0):
    n_keypoints = dense_keypoints0.shape[0]
    dense_keypoints = np.empty((n_keypoints, len(trackers) + 1, 2))
    dense_keypoints[:, 0] = dense_keypoints0
    for i, track01 in enumerate(trackers):
        dense_keypoints[:, i+1] = track01(dense_keypoints[:, i])
    return dense_keypoints


def dense_mvs(frames):
    features = [extract_features(f.image) for f in frames]

    trackers = []
    for i in range(len(frames)-1):
        features0, features1 = features[i], features[i+1]
        matches01 = match(features0, features1)
        tracker = Tracker(
            features0.keypoints[matches01[:, 0]],
            features1.keypoints[matches01[:, 1]],
            frames[i+1].image
        )
        trackers.append(tracker)

    dense_keypoints0 = extract_curvature_extrema(frames[0].image)

    dense_keypoints = track(trackers, dense_keypoints0)

    fig = plt.figure()
    for i in range(len(frames)):
        ax = fig.add_subplot(1, len(frames), i+1)
        ax.imshow(frames[i].image)
        ax.scatter(dense_keypoints[:, i, 0], dense_keypoints[:, i, 1],
                   s=0.1, c='red')
    plt.show()

    dense_keypoints = undistort(
        [f.camera_model for f in frames],
        dense_keypoints
    )

    poses = [Pose(f.rotation, f.position).world_to_local() for f in frames]
    points, depths = Triangulation(poses).triangulate(dense_keypoints)
    return poses, points, dense_keypoints


from tadataka.dataset.tum_rgbd import TumRgbdDataset
from tadataka.local_ba import run_ba

dataset = TumRgbdDataset(Path("datasets", "rgbd_dataset_freiburg1_xyz"))
poses, points, dense_keypoints = dense_mvs(dataset[270:274])

plot_map(poses, points)

point_indices, viewpoint_indices = np.where(np.ones((len(points), len(poses))))

# (n_points, n_poses, 2) -> (n_points * n_poses, 2)
dense_keypoints = dense_keypoints.reshape(-1, 2)

poses, points = run_ba(viewpoint_indices, point_indices,
                       poses, points, dense_keypoints)

plot_map(poses, points)

exit(0)


# dataset = NewTsukubaDataset(dataset_root)

frame0 = dataset[270]
frame1 = dataset[271]

image0 = frame0.image
sparse_triangulation(frame0, frame1)
dense_match_triangulation(frame0, frame1)
dense_track_triangulation(frame0, frame1)
