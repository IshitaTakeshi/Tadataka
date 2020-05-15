import numpy as np
cimport numpy as cnp


cdef extern from "_transform_project/_transform_project.h":
    void _transform_project(double *point, double *rotvec, double *t,
                            double *out);


cdef extern from "_transform_project/_pose_jacobian.h":
    void _pose_jacobian(double *point, double *rotvec, double *t, double *out);


cdef extern from "_transform_project/_point_jacobian.h":
    void _point_jacobian(double *point, double *rotvec, double *t, double *out);


cdef extern from "_transform_project/_exp_so3.h":
    void _exp_so3(double *rotvec, double *out);


def transform_project(cnp.ndarray[cnp.double_t, ndim=1] pose,
                      cnp.ndarray[cnp.double_t, ndim=1] point):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x;
    x = np.empty(2, dtype=np.float64)
    _transform_project(&point[0], &pose[0], &pose[3], &x[0])
    return x


def pose_jacobian(cnp.ndarray[cnp.double_t, ndim=1] pose,
                  cnp.ndarray[cnp.double_t, ndim=1] point):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] J;
    J = np.empty(12, dtype=np.float64)
    _pose_jacobian(&point[0], &pose[0], &pose[3], &J[0])
    return J.reshape(6, 2)


def point_jacobian(cnp.ndarray[cnp.double_t, ndim=1] pose,
                   cnp.ndarray[cnp.double_t, ndim=1] point):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] J;
    J = np.empty(6, dtype=np.float64)
    _point_jacobian(&point[0], &pose[0], &pose[3], &J[0])
    return J.reshape(3, 2)


def exp_so3(cnp.ndarray[cnp.double_t, ndim=1] rotvec):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] R;
    R = np.empty((3, 3), dtype=np.float64)
    _exp_so3(&rotvec[0], &R[0, 0]);
    return R
