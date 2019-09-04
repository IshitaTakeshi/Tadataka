from autograd import numpy as np

from skimage import transform as tf
from skimage.color import rgb2gray
from skimage.data import astronaut

from vitamine.visual_odometry.flow_estimation import estimate_affine_transform

from tests.utils import relative_error


def test_estimate_affine_transform():
    # test if affine parameters can be estimated correctly from
    # an image pair

    inv_transform_true = tf.AffineTransform(
        rotation=0.1,
        translation=[40.0, -20.0]
    )

    matrix = np.linalg.inv(inv_transform_true.params)

    A_true = matrix[0:2, 0:2]
    b_true = matrix[0:2, 2]

    image1 = rgb2gray(astronaut())
    image2 = tf.warp(image1, inv_transform_true)

    tform = estimate_affine_transform(image1, image2)

    A_pred, b_pred = tform.A, tform.b

    assert(relative_error(A_true.flatten(), A_pred.flatten()) < 0.05)
    assert(relative_error(b_true, b_pred) < 0.05)
