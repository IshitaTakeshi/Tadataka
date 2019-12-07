import numpy as np
from collections import deque

from skimage.color import rgb2gray

from tadataka.flow_estimation.image_curvature import (
    compute_image_curvature, extract_curvature_extrema
)
from tadataka.flow_estimation.flow_estimation import estimate_affine_transform
from tadataka.flow_estimation.extrema_tracker import ExtremaTracker
from tadataka.triangulation import Triangulation
from tadataka.utils import is_in_image_range
from tadataka.visual_odometry.feature_based import FeatureBasedVO


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

@property
def viewpoints(self):
    v = self.start_viewpoint
    return np.arange(v, v + self.window_size)


def compute_range_mask(keypoints, image_shape):
    mask = is_in_image_range(keypoints.reshape(-1, 2), image_shape)
    return mask.reshape(keypoints.shape[0:2])


def triangulate(poses, keypoints, keypoint_mask):
    assert(keypoints.shape[0:2] == keypoint_mask.shape[0:2])
    window_size, n_keypoints = keypoints.shape[0:2]
    points = np.empty((n_keypoints, 3))
    depth_mask = np.empty(n_keypoints, np.bool)
    for i in range(n_keypoints):
        viewpoint_indices = np.where(keypoint_mask[:, i])[0]

        # make two viewpoints different as possible
        # to get a larger parallax
        v1, v2 = np.min(viewpoint_indices), np.max(viewpoint_indices)

        t = Triangulation(poses[v1], poses[v2])
        keypoint1, keypoint2 = keypoints[v1, i], keypoints[v2, i]
        points[i], depth_mask[i] = t.triangulate_(keypoint1, keypoint2)
    return points, depth_mask


def compute_point_mask(keypoint_mask):
    # to perform triangulation,
    # we need to observe a 3D point from at least two viewpoints
    return np.sum(keypoint_mask, axis=0) >= 2


def compute_viewpoint_mask(keypoint_mask):
    return np.sum(keypoint_mask, axis=1) >= 1


def next_generator_id(ids):
    if len(ids) == 0:
        return 0
    return max(ids)


class KeypointFlow(object):
    def __init__(self, size):
        self.list = [None] * size
        self.size = size

    def flow(self, tracker):
        # shift values
        # [item0, item1, ..., itemN] -> [None, item0, ..., itemN-1]
        # return itemN

        out = self.list[-1]

        for i in reversed(range(1, self.size)):
            keypoints = self.list[i-1]
            if keypoints is None:
                continue
            keypoints[i] = tracker(keypoints[i-1])
            self.list[i] = keypoints

        return out

    def set(self, new_keypoints):
        # new_keypoints.shape == (n_keypoints, 2)
        keypoints = np.empty((self.size, new_keypoints.shape[0], 2))
        keypoints[0] = new_keypoints
        self.list[0] = keypoints


class VitaminE(FeatureBasedVO):
    def __init__(self, camera_model, window_size):
        super().__init__(camera_model)
        self.window_size = window_size
        self.keypoint_flow = KeypointFlow(self.window_size)
        self.points = np.empty((0, 3), np.float64)

    def get_tracker(self, image1):
        viewpoint0 = self.active_viewpoints[-2]
        viewpoint1 = self.active_viewpoints[-1]
        kd0, kd1 = self.kds[viewpoint0], self.kds[viewpoint1]
        return Tracker(self.matcher, kd0, kd1, image1)

    def add(self, image1):
        super().add(image1)

        keypoints = None
        if len(self.active_viewpoints) >= 2:
            tracker = self.get_tracker(image1)
            keypoints = self.keypoint_flow.flow(tracker)

        new_keypoints = extract_curvature_extrema(image1)
        self.keypoint_flow.set(new_keypoints)

        if keypoints is None:
            return self.active_viewpoints[-1]

        from tadataka.plot.visualizers import plot3d
        from tadataka.plot.common import axis3d
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(image1)
        keypoints_ = keypoints[-1]
        ax.scatter(keypoints_[:, 0], keypoints_[:, 1], s=0.1, c='red')
        plt.show()

        viewpoints = self.active_viewpoints[-self.window_size:]
        poses = [self.poses[v] for v in viewpoints]
        points = triangulate_(keypoints, poses, image1.shape)

        print("points shape", points.shape)
        ax = axis3d()
        plot3d(ax, points)
        plt.show()
        self.points = np.vstack((self.points, points))
        return self.active_viewpoints[-1]


def triangulate_(keypoints, poses, image_shape):
    keypoint_mask = compute_range_mask(keypoints, image_shape)
    point_mask = compute_point_mask(keypoint_mask)
    keypoints = keypoints[:, point_mask]
    keypoint_mask = keypoint_mask[:, point_mask]

    points, depth_mask = triangulate(poses, keypoints, keypoint_mask)
    return points # [depth_mask]
