import numpy as np
cimport numpy as cnp


cdef extern from "_radtan_distort_jacobian.h":
    void distort_jacobian(double k1, double k2, double k3,
                          double p1, double p2,
                          double x, double y, double *out)


cdef extern from "_radtan_distort.h":
    void distort(double k1, double k2, double k3,
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

    cdef cnp.ndarray[cnp.float64_t, ndim=1] out

    out = np.empty(4, dtype=np.float64)

    distort_jacobian(k1, k2, k3, p1, p2, x, y, &out[0])
    return out.reshape((2, 2))


def distort_(cnp.ndarray[cnp.double_t, ndim=1] keypoint,
             cnp.ndarray[cnp.double_t, ndim=1] dist_coeffs):
    cdef double x = keypoint[0]
    cdef double y = keypoint[1]
    cdef double k1 = dist_coeffs[0]
    cdef double k2 = dist_coeffs[1]
    cdef double p1 = dist_coeffs[2]
    cdef double p2 = dist_coeffs[3]
    cdef double k3 = dist_coeffs[4]

    cdef cnp.ndarray[cnp.float64_t, ndim=1] out

    out = np.empty(2, dtype=np.float64)

    distort(k1, k2, k3, p1, p2, x, y, &out[0])
    return out


def radtan_distort(cnp.ndarray[cnp.double_t, ndim=2] keypoints,
                   cnp.ndarray[cnp.double_t, ndim=1] dist_coeffs):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out

    out = np.empty((keypoints.shape[0], keypoints.shape[1]), dtype=np.float64)
    for i in range(keypoints.shape[0]):
        out[i] = distort_(keypoints[i], dist_coeffs)
    return out


def inv2x2(cnp.ndarray[cnp.double_t, ndim=2] X):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out
    out = np.empty((2, 2), dtype=np.float64)

    a, b, c, d = X[0, 0], X[0, 1], X[1, 0], X[1, 1]
    det = a * d - b * c
    out[0, 0] = d
    out[0, 1] = -b
    out[1, 0] = -c
    out[1, 1] = a
    out = out / det
    return out


def undistort_(cnp.ndarray[cnp.double_t, ndim=1] keypoint,
               cnp.ndarray[cnp.double_t, ndim=1] dist_coeffs,
               int max_iter, float threshold):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] J
    cdef cnp.ndarray[cnp.float64_t, ndim=1] r, d
    cdef cnp.ndarray[cnp.float64_t, ndim=1] p = np.copy(keypoint)

    for i in range(max_iter):
        J = radtan_distort_jacobian(p, dist_coeffs)
        r = keypoint - distort_(p, dist_coeffs)
        # d = inv(J.T * J) * J.T * r
        # J.T * J * d = J.T * r
        # J * d = r
        # if J is a square matrix
        # d = inv(J) * r
        d = np.dot(inv2x2(J), r)

        if np.dot(d, d) < threshold:
            break

        p = p + d
    return p


def radtan_undistort(cnp.ndarray[cnp.double_t, ndim=2] keypoints,
                     cnp.ndarray[cnp.double_t, ndim=1] dist_coeffs,
                     int max_iter, float threshold):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out

    out = np.empty((keypoints.shape[0], keypoints.shape[1]), dtype=np.float64)
    for i in range(keypoints.shape[0]):
        out[i] = undistort_(keypoints[i], dist_coeffs, max_iter, threshold)
    return out
