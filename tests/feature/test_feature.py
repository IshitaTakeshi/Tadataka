import numpy as np
from tadataka.feature.feature import extract_keypoints


def test_extract_keypoints():
    height, width = 200, 100
    image = np.random.randint(0, 256, (height, width))
    keypoints = extract_keypoints(image)

    assert(keypoints.shape[1] == 2)

    xs = keypoints[:, 0]
    ys = keypoints[:, 1]

    mask_x = np.logical_and(0 <= xs, xs < width)
    mask_y = np.logical_and(0 <= ys, ys < height)
    assert(np.logical_and(mask_x, mask_y).all())
