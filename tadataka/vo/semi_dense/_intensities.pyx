cimport numpy as cnp
from numpy.math cimport INFINITY


def search_(cnp.ndarray[cnp.float64_t, ndim=1] sequence,
            cnp.ndarray[cnp.float64_t, ndim=1] kernel):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] d;
    cdef double error;

    cdef double min_error = INFINITY
    cdef int N = len(kernel)
    cdef int argmin = -1

    for i in range(len(sequence)-N+1):
        d = sequence[i:i+N] - kernel
        error = (d * d).sum()
        if error < min_error:
            min_error = error
            argmin = i
    return argmin


def search_intensities(cnp.ndarray[cnp.float64_t, ndim=1] intensities_key,
                       cnp.ndarray[cnp.float64_t, ndim=1] intensities_ref):
    cdef int argmin = search_(intensities_ref, intensities_key)
    cdef int offset = len(intensities_key) // 2
    return argmin + offset
