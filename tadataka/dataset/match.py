import numpy as np

from skimage.feature import match_descriptors


def match_timestamps(timestamps0, timestamps1, max_difference):
    # Regard timestamps as 1D float desciptors
    matches01 = match_descriptors(timestamps0.reshape(-1, 1),
                                  timestamps1.reshape(-1, 1),
                                  cross_check=True)
    diff = np.abs(timestamps0[matches01[:, 0]] - timestamps1[matches01[:, 1]])
    return matches01[diff <= max_difference]
