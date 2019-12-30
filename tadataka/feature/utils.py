from numba import njit


@njit
def mask_border_keypoints(image_shape, keypoints, distance):
    height, width = image_shape[0:2]
    return ((distance <= keypoints[:, 0]) &
            (distance <= keypoints[:, 1]) &
            (height - distance > keypoints[:, 1]) &
            (width - distance > keypoints[:, 0]))
