import numpy as np
from skimage import exposure
from skimage.color import rgb2gray
import pandas as pd

from tadataka.feature import Matcher
from tadataka.flow_estimation.image_curvature import (
    extract_curvature_extrema, compute_image_curvature)
from tadataka.flow_estimation.flow_estimation import estimate_affine_transform
from tadataka.flow_estimation.extrema_tracker import ExtremaTracker
from tadataka.triangulation import TwoViewTriangulation
from tadataka.utils import is_in_image_range


match = Matcher(enable_ransac=True, enable_homography_filter=True)


class DenseKeypointExtractor(object):
    def __init__(self, percentile):
        self.percentile = percentile

    def __call__(self, image):
        return extract_curvature_extrema(image, self.percentile)


extract_dense_keypoints = DenseKeypointExtractor(percentile=98)


def keypoints_from_new_area(image1, flow01):
    """Extract keypoints from newly observed image area"""
    keypoints1 = extract_dense_keypoints(image1)
    # out of image range after transforming from frame1 to frame0
    # we assume image1.shape == image0.shape
    mask = ~is_in_image_range(flow01.inverse(keypoints1), image1.shape)
    return keypoints1[mask]


def normalize_image(image):
    image = rgb2gray(image)
    image = exposure.equalize_adapthist(image)
    return image


def track_(keypoints0, image1, flow01, lambda_):
    image1 = normalize_image(image1)
    curvature = compute_image_curvature(image1)

    tracker = ExtremaTracker(curvature, lambda_)
    return tracker.optimize(flow01(keypoints0))


def estimate_flow(features0, features1):
    matches01 = match(features0, features1)
    keypoints0 = features0.keypoints[matches01[:, 0]]
    keypoints1 = features1.keypoints[matches01[:, 1]]
    return estimate_affine_transform(keypoints0, keypoints1)


class Tracker(object):
    def __init__(self, flow01, image1, lambda_):
        self.flow01 = flow01
        self.image1 = image1
        self.lambda_ = lambda_

    def __call__(self, keypoints0):
        # track keypoints
        keypoints0_ = get_array(keypoints0)
        keypoints1_ = track_(keypoints0_,
                             self.image1, self.flow01, self.lambda_)
        mask1 = is_in_image_range(keypoints1_, self.image1.shape)
        ids0 = get_ids(keypoints0)
        keypoints1 = create_keypoint_frame_(ids0[mask1], keypoints1_[mask1])

        # keypoints extracted from the newly observed image area
        id_start = ids0[-1] + 1  # assign new indices
        new_keypoints1 = keypoints_from_new_area(self.image1, self.flow01)
        new_rows = create_keypoint_frame(id_start, new_keypoints1)

        return pd.concat([keypoints1, new_rows])


def match_keypoint_ids(ids0, ids1):
    _, indices0, indices1 = np.intersect1d(ids0, ids1, return_indices=True)
    return np.column_stack((indices0, indices1))


def match_keypoints(keypoints0, keypoints1):
    return match_keypoint_ids(get_ids(keypoints0), get_ids(keypoints1))


def match_multiple_keypoint_ids(keypoint_ids):
    from functools import reduce
    # ids = [get_ids(keypoints) for keypoints_ in keypoints]
    shared_ids = reduce(np.intersect1d, keypoint_ids)
    matches = np.empty((len(shared_ids), len(keypoint_ids)), dtype=np.int64)
    for i, ids in enumerate(keypoint_ids):
        matches01 = match_keypoint_ids(shared_ids, ids)
        matches[:, i] = matches01[:, 1]
    return matches


def match_multiple_keypoints(keypoints):
    return match_multiple_keypoint_ids(
        [get_ids(keypoints_) for keypoints_ in keypoints]
    )


def init_keypoint_frame(image):
    keypoints = extract_dense_keypoints(image)
    return create_keypoint_frame(0, keypoints)


def create_keypoint_frame(start_id, keypoints):
    N = keypoints.shape[0]
    ids = np.arange(start_id, start_id + N)
    return create_keypoint_frame_(ids, keypoints)


def create_keypoint_frame_(ids, keypoints):
    assert(keypoints.shape == (ids.shape[0], 2))
    return pd.DataFrame({'id': ids,
                         'x': keypoints[:, 0],
                         'y': keypoints[:, 1]})


def get_array(frame):
    return frame[['x', 'y']].to_numpy()


def get_ids(frame):
    return frame['id'].to_numpy()
