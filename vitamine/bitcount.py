import numpy as np
from numba import njit

m1  = 0x55  # binary: 01010101
m2  = 0x33  # binary: 00110011
m4  = 0x0f  # binary: 00001111

# @njit
def popcount8a(x):
    x = (x & m1) + ((x >> 1) & m1)
    x = (x & m2) + ((x >> 2) & m2)
    x = (x & m4) + ((x >> 4) & m4)
    return x


@njit
def popcount8b(x):
    x -= (x >> 1) & m1
    x = (x & m2) + ((x >> 2) & m2)
    x = (x + (x >> 4)) & m4
    return x


@njit
def popcount(x):
    y = (x & m1) + ((x >> 1) & m1)
    z = (y & m2) + ((y >> 2) & m2)
    return (z + (z >> 4)) & m4


@njit
def bitdistances(A, B):
    N = A.shape[0]
    M = B.shape[0]

    D = np.empty((N, M), dtype=np.uint64)
    for i in range(N):
        for j in range(M):
            x = np.bitwise_xor(A[i], B[j])
            D[i, j] = np.sum(popcount(x))
    return D


def distances(A, B):
    A = np.packbits(A, axis=1)
    B = np.packbits(B, axis=1)
    return bitdistances(A, B)
