import numpy as np


def normalize_length(v):
    # TODO avoid division by zero?
    return v / np.linalg.norm(v)
