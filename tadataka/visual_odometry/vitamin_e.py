import numpy as np

from skimage.color import rgb2gray

from tadataka.flow_estimation.image_curvature import (
    compute_image_curvature, extract_curvature_extrema
)
from tadataka.flow_estimation.flow_estimation import estimate_affine_transform
from tadataka.flow_estimation.extrema_tracker import ExtremaTracker
from tadataka.utils import is_in_image_range
from tadataka.visual_odometry.visual_odometry import FeatureBasedVO


def keypoint_correction(keypoints, image):
    curvature = compute_image_curvature(rgb2gray(image))
    tracker = ExtremaTracker(curvature, lambda_=1e-3)
    mask = is_in_image_range(keypoints, image.shape)
    keypoints[mask] = tracker.optimize(keypoints[mask])
    return keypoints


class Tracker(object):
    def __init__(self, matcher, kd0, kd1, image1):
        self.kd0, self.kd1 = kd0, kd1
        self.image1 = image1
        matches01 = matcher(kd0, kd1)
        self.flow = estimate_affine_transform(kd0.keypoints[matches01[:, 0]],
                                              kd1.keypoints[matches01[:, 1]])

    def __call__(self, dense_keypoints0):
        dense_keypoints1 = self.flow.transform(dense_keypoints0)
        dense_keypoints1 = keypoint_correction(dense_keypoints1, self.image1)
        return dense_keypoints1


class DenseKeypointsGenerator(object):
    def __init__(self, image, window_size):
        keypoints = extract_curvature_extrema(image, percentile=99)

        # from matplotlib import pyplot as plt
        # plt.imshow(image)
        # plt.scatter(keypoints[:, 0], keypoints[:, 1], s=0.1, c='red')
        # plt.show()

        self.keypoints = np.empty((window_size, *keypoints.shape))
        self.keypoints[0] = keypoints
        self.index = 0

    def reached_max(self):
        return self.index + 1 >= self.keypoints.shape[0]

    def track(self, tracker):
        self.keypoints[self.index+1] = tracker(self.keypoints[self.index])
        self.index += 1

    def export(self):
        # Check if each feature point exists in the image range
        # in at least two frames.
        # If the keypoint is in image range in only one frame,
        # it cannot be triangulated.

        return self.keypoints

@property
def viewpoints(self):
    v = self.start_viewpoint
    return np.arange(v, v + self.window_size)


def compute_range_mask(keypoints):
    mask = is_image_range(keypoints.reshape(-1, 2))
    return mask.reshape(keypoints.shape[0:2])


def triangulate(poses, keypoints, keypoint_mask):
    assert(keypoints.shape == keypoint_mask.shape)
    points = np.empty((n_keypoints, 3))

    for i in range(n_keypoints):
        viewpoint_indices = np.where(keypoint_mask[:, i])[0]

        # make two viewpoints different as possible
        # to get a larger parallax
        v1, v2 = np.min(viewpoint_indices), np.max(viewpoint_indices)

        t = Triangulation(poses[v1], poses[v2])
        points[i] = t.triangulate(keypoints[v1, i], keypoints[v2, i])

    return points


def compute_triangulation_mask(keypoint_mask):
    # to perform triangulation,
    # we need to observe a 3D point from at least two viewpoints
    return np.sum(keypoint_mask, axis=1) >= 2


def compute_viewpoint_mask(keypoint_mask):
    return np.sum(keypoint_mask, axis=0) >= 1


def next_generator_id(ids):
    if len(ids) == 0:
        return 0
    return max(ids)


def track(tracker, keypoint_generators):
    arg_reached = []
    for i, generator in enumerate(keypoint_generators):
        generator.track(tracker)

        if not generator.reached_max():
            arg_reached.append(i)
            continue

    return keypoint_generators, arg_reached


class VitaminE(FeatureBasedVO):
    def __init__(self, camera_parameters, distortion_model, window_size):
        super().__init__(camera_parameters, distortion_model)
        self.__window_size = window_size
        self.keypoint_generators = []

    def add(self, image1):
        super().add(image1)

        if len(self.active_viewpoints) >= 2:
            # track existing
            viewpoint0 = self.active_viewpoints[-2]
            viewpoint1 = self.active_viewpoints[-1]
            kd0, kd1 = self.kds[viewpoint0], self.kds[viewpoint1]
            tracker = Tracker(self.matcher, kd0, kd1, image1)
            self.keypoint_generators, indices_to_remove = track(
                tracker, self.keypoint_generators
            )

            for i in sorted(indices_to_remove, reverse=True):
                del self.keypoint_generators[i]

        # add new
        g = DenseKeypointsGenerator(image1, self.__window_size)
        self.keypoint_generators.append(g)
        return self.active_viewpoints[-1]


def triangulate(keypoints, poses):
    keypoint_mask = compute_range_mask(keypoints)
    triangulation_mask = compute_triangulation_mask(keypoint_mask)
    viewpoint_mask = compute_viewpoint_mask(keypoint_mask)

    keypoints = keypoints[viewpoint_mask, triangulation_mask]
    keypoint_mask = keypoint_mask[viewpoint_mask, keypoint_mask]

    poses_ = poses[viewpoint_mask]

    points = triangulate(poses_, keypoints, keypoint_mask)

    viewpoint_indices, point_indices = np.where(keypoint_mask)

    poses_, points = try_run_ba(viewpoint_indices, point_indices,
                                poses_, points, keypoints[keypoint_mask])

    poses[viewpoint_mask] = poses_
    return poses, points
