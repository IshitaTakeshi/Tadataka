from numpy.testing import assert_array_equal
from tadataka.coordinates import image_coordinates

def test_image_coordinates():
    width, height = 3, 4
    coordinates = image_coordinates(image_shape=[height, width])
    assert_array_equal(
        coordinates,
#         x  y
        [[0, 0],
         [1, 0],
         [2, 0],
         [0, 1],
         [1, 1],
         [2, 1],
         [0, 2],
         [1, 2],
         [2, 2],
         [0, 3],
         [1, 3],
         [2, 3]]
    )
