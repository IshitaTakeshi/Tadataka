import numpy as np
cimport numpy as cnp


cdef extern from "_radtan_distort_jacobian.h":
    void distort_jacobian(double k1, double k2, double k3,
                          double p1, double p2,
                          double x, double y, double *out)


def radtan_distort_jacobian(cnp.ndarray[cnp.double_t, ndim=1] keypoint,
                            cnp.ndarray[cnp.double_t, ndim=1] dist_coeffs):
    cdef double x = keypoint[0]
    cdef double y = keypoint[1]
    cdef double k1 = dist_coeffs[0]
    cdef double k2 = dist_coeffs[1]
    cdef double p1 = dist_coeffs[2]
    cdef double p2 = dist_coeffs[3]
    cdef double k3 = dist_coeffs[4]

    out = np.empty(4, dtype=np.double)
    cdef double[::1] out_memview = out

    distort_jacobian(k1, k2, k3, p1, p2, x, y, &out_memview[0])
    return out.reshape((2, 2))
