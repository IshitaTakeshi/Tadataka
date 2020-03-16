from numba import njit

@njit
def invert_depth(depth, EPSILON=1e-16):
    return 1 / (depth + EPSILON)
