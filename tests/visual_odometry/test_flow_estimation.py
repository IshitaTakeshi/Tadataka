from autograd import numpy as np

from skimage import transform as tf
from skimage.color import rgb2gray
from skimage.data import astronaut

from vitamine.visual_odometry.flow_estimation import AffineTransformEstimator


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

    estimator = AffineTransformEstimator()
    affine_transform = estimator.estimate(image1, image2)

    A_pred, b_pred = affine_transform.A, affine_transform.b

    # test relative error
    threshold = np.linalg.norm(A_true.flatten()) * 0.05
    assert(np.linalg.norm((A_true-A_pred).flatten()) < threshold)

    threshold = np.linalg.norm(b_true.flatten()) * 0.05
    assert(np.linalg.norm(b_true-b_pred) < threshold)
