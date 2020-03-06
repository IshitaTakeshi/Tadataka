import numpy as np
import numba

@numba.njit
def normalize_length(v):
    return v / np.linalg.norm(v)
