from autograd import numpy as np

from numpy.testing import assert_array_almost_equal

from skimage import transform as tf
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.data import astronaut

from vitamine.visual_odometry.triangulation import match_affine, triangulation


def test_triangulation():
    K = np.array([
        [1520.4, 0.0, 302.32],
        [0.0, 1525.9, 246.87],
        [0.0, 0.0, 1.0]
    ])

    # TODO download the temple dataset
    image1 = imread("./datasets/templeRing/templeR0003.png")
    image2 = imread("./datasets/templeRing/templeR0005.png")

    points, local_maximums1, local_maximums2 = triangulation(
        rgb2gray(image1), rgb2gray(image2), K)
    print("points")
    print(points)


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

    affine_transform, keypoints1, keypoints2 = match_affine(image1, image2)

    A_pred, b_pred = affine_transform.A, affine_transform.b

    # test relative error
    threshold = np.linalg.norm(A_true.flatten()) * 0.05
    assert(np.linalg.norm((A_true-A_pred).flatten()) < threshold)

    threshold = np.linalg.norm(b_true.flatten()) * 0.05
    assert(np.linalg.norm(b_true-b_pred) < threshold)
