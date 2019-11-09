import numpy as np
from numba import njit


@njit
def popcount(x):
    y = (x & 0x55) + ((x >> 1) & 0x55)
    z = (y & 0x33) + ((y >> 2) & 0x33)
    return (z + (z >> 4)) & 0x0f


@njit(parallel=True)
def bitdistances(A, B):
    # compute hamming distances between packed bit arrays
    N = A.shape[0]
    M = B.shape[0]

    D = np.empty((N, M), dtype=np.uint64)
    for i in range(N):
        for j in range(M):
            x = np.bitwise_xor(A[i], B[j])
            # x is an uint8 array
            # count 1 bits in each element of x and merge
            D[i, j] = np.sum(popcount(x))
    return D


def distances(A, B):
    # compute hamming distances between bool arrays
    A = np.packbits(A, axis=1)
    B = np.packbits(B, axis=1)
    return bitdistances(A, B)
