import warnings
import numpy as np


def warn_points_behind_cameras():
    message = "Most of points are behind cameras. Maybe wrong matches?"
    warnings.warn(message, RuntimeWarning)


def depth_condition(depth_mask, positive_depth_ratio=0.8):
    # number of positive depths / total number of keypoints
    # (or corresponding points)
    return np.sum(depth_mask) / len(depth_mask) >= positive_depth_ratio


def compute_depth_mask(depths, min_depth=0.0):
    return np.all(depths > min_depth, axis=0)
