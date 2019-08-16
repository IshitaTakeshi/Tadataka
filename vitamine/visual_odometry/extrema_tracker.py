from autograd import numpy as np

from skimage.feature import peak_local_max

from vitamine.coordinates import yx_to_xy
from vitamine.flow_estimation.extrema_tracker import ExtremaTracker
from vitamine.flow_estimation.image_curvature import compute_image_curvature
from vitamine.utils import round_int, is_in_image_range
from vitamine.visual_odometry.flow_estimation import AffineTransformEstimator


def extract_local_maximums(curvature):
    local_maximums = peak_local_max(curvature, min_distance=5)
    return yx_to_xy(local_maximums)


def extrema_tracking(curvature1, curvature2, affine_transform, lambda_):
    image_shape = curvature1.shape[0:2]

    local_maximums1 = extract_local_maximums(curvature1)

    local_maximums2 = affine_transform.transform(local_maximums1)
    local_maximums2 = round_int(local_maximums2)

    mask = is_in_image_range(local_maximums2, image_shape)

    # filter local maximums so that all of them fit in the image range
    # after affine transform
    local_maximums1 = local_maximums1[mask]
    local_maximums2 = local_maximums2[mask]

    tracker = ExtremaTracker(curvature2, local_maximums2, lambda_)
    local_maximums2 = tracker.optimize()

    assert(is_in_image_range(local_maximums2, image_shape).all())

    return local_maximums1, local_maximums2


class TwoViewExtremaTracker(object):
    def __init__(self, affine_transform, curvature1, lambda_):
        self.image_shape = curvature1.shape[0:2]
        self.curvature1 = curvature1
        self.lambda_ = lambda_
        self.affine = affine_transform

    def track(self, local_maximums):
        local_maximums = self.affine.transform(local_maximums)
        local_maximums = round_int(local_maximums)

        # correct the coordinate if local maximum is in the image range
        # leave it otherwise
        mask = is_in_image_range(local_maximums, self.image_shape)

        tracker = ExtremaTracker(self.curvature1, local_maximums[mask],
                                 self.lambda_)
        local_maximums[mask] = tracker.optimize()

        return local_maximums


def propagate(local_maximums, affines, curvatures, lambda_):
    assert(len(affines) == len(curvatures))

    image_shape = curvatures[0].shape[0:2]

    N = len(curvatures)

    L = np.full((N+1, *local_maximums.shape), np.nan)
    L[0] = local_maximums

    for i in range(N):
        # note that local_maximums is always non nan
        tracker = TwoViewExtremaTracker(affines[i], curvatures[i], lambda_)

        local_maximums = tracker.track(local_maximums)

        mask = is_in_image_range(local_maximums, image_shape)

        L[i+1, mask] = local_maximums[mask]
    return L


class MultipleViewExtremaTracker(object):
    def __init__(self, images, lambda_=0.1):
        self.images = images
        self.image_shape = self.images[0].shape

        self.curvatures = [compute_image_curvature(I) for I in images]
        self.lambda_ = lambda_

    def track(self):
        images = self.images

        N = len(images)
        e = AffineTransformEstimator()
        affines = [e.estimate(images[i], images[i+1]) for i in range(0, N-1)]

        local_maximums = extract_local_maximums(self.curvatures[0])

        return propagate(local_maximums, affines,
                         self.curvatures[1:], self.lambda_)
