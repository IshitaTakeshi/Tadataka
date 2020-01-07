import numpy as np
from numba import njit

from tadataka.feature.utils import mask_border_keypoints


@njit
def extract_patch(image, p, patch_size):
    s = patch_size // 2
    x, y = p
    return image[y-s:y+s+1, x-s:x+s+1]


@njit
def extract_patches(image, keypoints, patch_size):
    assert(keypoints.shape[1] == 2)
    assert(patch_size % 2 == 1)

    N = keypoints.shape[0]

    patches = np.empty((N, patch_size, patch_size))
    for i in range(N):
        patches[i] = extract_patch(image, keypoints[i], patch_size)
    return patches
