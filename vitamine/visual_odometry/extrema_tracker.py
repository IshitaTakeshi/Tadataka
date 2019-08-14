from skimage.feature import peak_local_max
from vitamine.coordinates import yx_to_xy

from vitamine.utils import round_int, is_in_image_range
from vitamine.flow_estimation.extrema_tracker import ExtremaTracker


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
