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


def extrema_tracking(curvature1, curvature2, affine_transform):
    image_shape = curvature1.shape[0:2]

    local_maximums1 = extract_local_maximums(curvature1)

    local_maximums2 = affine_transform.transform(local_maximums1)
    local_maximums2 = round_int(local_maximums2)

    mask = is_in_image_range(local_maximums2, image_shape)

    # filter local maximums so that all of them fit in the image range
    # after affine transform
    local_maximums1 = local_maximums1[mask]
    local_maximums2 = local_maximums2[mask]

    tracker = ExtremaTracker(curvature2, local_maximums2, lambda_=0.1)
    local_maximums2 = tracker.optimize()

    assert(is_in_image_range(local_maximums2, image_shape).all())

    return local_maximums1, local_maximums2


class MultipleViewExtremaTracker(object):
    def __init__(self, images, lambda_=0.1):
        self.images = images
        self.image_shape = self.images[0].shape

        self.curvatures = [compute_image_curvature(I) for I in images]
        self.lambda_ = lambda_

    def estimate_next(self, local_maximums, image1, image2, curvature2):
        estimator = AffineTransformEstimator()
        affine = estimator.estimate(image1, image2)

        local_maximums = affine.transform(local_maximums)
        local_maximums = round_int(local_maximums)

        # correct the coordinate if local maximum is in the image range
        # leave it otherwise
        mask = is_in_image_range(local_maximums, self.image_shape)
        tracker = ExtremaTracker(curvature2, local_maximums[mask], self.lambda_)
        local_maximums[mask] = tracker.optimize()

        return local_maximums

    def track(self):
        n_images = len(self.images)

        local_maximums = extract_local_maximums(self.curvatures[0])

        L = np.full((n_images, *local_maximums.shape), np.nan)
        L[0] = local_maximums

        for i in range(0, n_images-1):
            local_maximums = self.estimate_next(
                local_maximums,
                self.images[i],
                self.images[i+1],
                self.curvatures[i+1]
            )

            mask = is_in_image_range(local_maximums, self.image_shape)

            L[i+1, mask] = local_maximums[mask]
        return L
