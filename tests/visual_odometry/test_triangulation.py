from autograd import numpy as np

from numpy.testing import assert_array_almost_equal

from skimage.io import imread
from skimage.color import rgb2gray

from vitamine.visual_odometry.triangulation import triangulation


def test_triangulation():
    K = np.array([
        [1520.4, 0.0, 302.32],
        [0.0, 1525.9, 246.87],
        [0.0, 0.0, 1.0]
    ])

    # TODO download the temple dataset
    image1 = imread("./datasets/templeRing/templeR0003.png")
    image2 = imread("./datasets/templeRing/templeR0005.png")

    points = triangulation(rgb2gray(image1), rgb2gray(image2), K)
    print("points")
    print(points)
