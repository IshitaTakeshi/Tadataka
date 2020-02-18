# reference
# https://github.com/colmap/colmap/blob/master/src/base/camera_models.h

import numpy as np
from autograd import jacobian
from autograd import numpy as anp

EPSILON = 1e-4


def fov_distort_factors(X, omega):
    def f(r):
        return np.arctan(2 * r * np.tan(omega / 2)) / omega

    r = np.linalg.norm(X, axis=1)

    iszero = np.isclose(r, 0)

    factors = np.empty(r.shape)
    factors[iszero] = 2 * np.tan(omega / 2) / omega  # linear approx of f
    factors[~iszero] = f(r[~iszero])
    return factors


def fov_undistort_factors(X, omega):
    def f(r):
        return np.tan(r * omega) / (2 * r * np.tan(omega / 2))

    r = np.linalg.norm(X, axis=1)

    iszero = np.isclose(r, 0)

    factors = np.empty(r.shape)
    factors[iszero] = omega / (2 * np.tan(omega / 2))  # linear approx of f
    factors[~iszero] = f(r[~iszero])
    return factors


class BaseDistortion(object):
    def __eq__(self, other):
        C1 = type(self) == type(other)
        C2 = np.isclose(self.params, other.params).all()
        return C1 and C2


class FOV(BaseDistortion):
    """
    Devernay, Frederic, and Olivier D. Faugeras.
    "Automatic calibration and removal of distortion from
     scenes of structured environments."
    Investigative and Trial Image Processing. Vol. 2567.
    International Society for Optics and Photonics, 1995.
    """
    def __init__(self, omega):
        self.omega = omega

    def distort(self, undistorted_keypoints):
        if np.isclose(self.omega, 0):
            return undistorted_keypoints

        factors = fov_distort_factors(undistorted_keypoints, self.omega)
        return factors.reshape(-1, 1) * undistorted_keypoints

    def undistort(self, distorted_keypoints):
        if np.isclose(self.omega, 0):
            return distorted_keypoints  # all factors = 1

        factors = fov_undistort_factors(distorted_keypoints, self.omega)
        return factors.reshape(-1, 1) * distorted_keypoints

    @staticmethod
    def from_params(params):
        assert(len(params) == 1)
        return FOV(omega=params[0])

    @property
    def params(self):
        return [self.omega]


class RadTan(BaseDistortion):
    def __init__(self, dist_coeffs):
        assert(len(dist_coeffs) <= 5)

        self.dist_coeffs = [0] * 5
        self.dist_coeffs[:len(dist_coeffs)] = dist_coeffs

    def _distort(self, p):
        k1, k2, p1, p2, k3 = self.dist_coeffs

        r2 = anp.sum(anp.power(p, 2))
        r4 = anp.power(r2, 2)
        r6 = anp.power(r2, 3)
        kr = 1 + k1 * r2 + k2 * r4 + k3 * r6

        x, y = p
        return anp.array([
            x * kr + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            y * kr + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
        ])

    def distort(self, undistorted_keypoints):
        N = undistorted_keypoints.shape[0]
        Q = np.empty(undistorted_keypoints.shape)
        for i in range(N):
            Q[i] = self._distort(undistorted_keypoints[i])
        return Q

    def _undistort(self, q, max_iter=100, threshold=1e-10):
        p = anp.array(q)
        for i in range(max_iter):
            J = jacobian(self._distort)(p)
            r = q - self._distort(p)
            d = np.linalg.solve(J, r)

            if np.dot(d, d) < threshold:
                break

            p = p + d
        return p

    def undistort(self, distorted_keypoints):
        N = distorted_keypoints.shape[0]
        P = np.empty(distorted_keypoints.shape)
        for i in range(N):
            P[i] = self._undistort(distorted_keypoints[i])
        return P

    @staticmethod
    def from_params(params):
        return RadTan(params)

    @property
    def params(self):
        return self.dist_coeffs
