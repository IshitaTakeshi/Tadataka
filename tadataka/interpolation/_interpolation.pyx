import numpy as np
cimport numpy as cnp


cdef extern from "_bilinear.h":
    void _interpolation(
        const double *image, const int image_width,
        const double *coordinates, const int n_coordinates,
        double *intensities);


def _interpolation_pyx(cnp.ndarray[cnp.float64_t, ndim=2] image,
                       cnp.ndarray[cnp.float64_t, ndim=2] coordinates):
    cdef int N = coordinates.shape[0]

    cdef double[:] image_view = image.flatten()
    cdef int width = image.shape[1]

    cdef double[:] coordinates_view = coordinates.flatten()

    intensities = np.empty(N, dtype=np.float64)

    cdef double[:] intensities_view = intensities
    _interpolation(&image_view[0], width, &coordinates_view[0], N,
                   &intensities_view[0])
    return intensities
