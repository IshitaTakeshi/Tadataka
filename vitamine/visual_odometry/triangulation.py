from autograd import numpy as np

from vitamine.bundle_adjustment.triangulation import points_from_unknown_poses
from vitamine.flow_estimation.image_curvature import compute_image_curvature
from vitamine.visual_odometry.extrema_tracker import extrema_tracking
from vitamine.visual_odometry.flow_estimation import estimate_affine_transform


def triangulation(image1, image2, K):
    assert(np.ndim(image1) == 2)
    assert(np.ndim(image2) == 2)

    affine_transform = estimate_affine_transform(image1, image2)

    local_maximums1, local_maximums2 = extrema_tracking(
        compute_image_curvature(image1),
        compute_image_curvature(image2),
        affine_transform
    )

    R, t, points = points_from_unknown_poses(
        local_maximums1,
        local_maximums2,
        K
    )

    return points
