# This is a modified version of IRLS implementation in 'sattsmodels'

# Copyright (C) 2006, Jonathan E. Taylor
# All rights reserved.
#
# Copyright (c) 2006-2008 Scipy Developers.
# All rights reserved.
#
# Copyright (c) 2009-2018 statsmodels Developers.
# All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of statsmodels nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL STATSMODELS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

import numpy as np
from scipy.stats import norm as Gaussian
import warnings


def mad(a, c=Gaussian.ppf(3/4.), axis=0):
    # c \approx .6745
    """
    The Median Absolute Deviation along given axis of an array

    Parameters
    ----------
    a : array_like
        Input array.
    c : float, optional
        The normalization constant.  Defined as scipy.stats.norm.ppf(3/4.),
        which is approximately .6745.
    Returns
    -------
    mad : float
        `mad` = median(abs(`a` - center))/`c`
    """
    # a = array_like(a, 'a', ndim=None)
    # c = float_like(c, 'c')
    return np.median(np.abs(a) / c, axis=axis)


def _estimate_scale(residual):
    return mad(residual)


class HuberT(object):
    """
    Huber's T for M estimation.

    Parameters
    ----------
    t : float, optional
        The tuning constant for Huber's t function. The default value is
        1.345.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, t=1.345):
        self.t = t

    def _subset(self, z):
        """
        Huber's T is defined piecewise over the range for z
        """
        z = np.asarray(z)
        return np.less_equal(np.abs(z), self.t)

    def rho(self, z):
        r"""
        The robust criterion function for Huber's t.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : array
            rho(z) = .5*z**2            for \|z\| <= t

            rho(z) = \|z\|*t - .5*t**2    for \|z\| > t
        """
        z = np.asarray(z)
        test = self._subset(z)
        return (test * 0.5 * z**2 +
                (1 - test) * (np.abs(z) * self.t - 0.5 * self.t**2))

    def psi(self, z):
        r"""
        The psi function for Huber's t estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : array
            psi(z) = z      for \|z\| <= t

            psi(z) = sign(z)*t for \|z\| > t
        """
        z = np.asarray(z)
        test = self._subset(z)
        return test * z + (1 - test) * self.t * np.sign(z)

    def weights(self, z):
        r"""
        Huber's t weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : array
            weights(z) = 1          for \|z\| <= t

            weights(z) = t/\|z\|      for \|z\| > t
        """
        z = np.asarray(z)
        test = self._subset(z)
        absz = np.abs(z)
        absz[test] = 1.0
        return test + (1 - test) * self.t / absz

    def psi_deriv(self, z):
        """
        The derivative of Huber's t psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        return np.less_equal(np.abs(z), self.t)

    def __call__(self, z):
        """
        Returns the value of estimator rho applied to an input
        """
        return self.rho(z)


class Residual(object):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def compute(self, params):
        return self.y - self.X.dot(params)


def least_squares(X, y):
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=-1)
    return params


def weighted_least_squares(X, y, weights):
    sqrt_weights = np.sqrt(weights)
    params, _, _, _ = np.linalg.lstsq(sqrt_weights[:, None] * X,
                                      sqrt_weights * y,
                                      rcond=-1)
    return params


def fit(X, y, max_iter=100, M=HuberT()):
    residual = Residual(X, y)
    params = least_squares(X, y)
    r = residual.compute(params)
    scale = _estimate_scale(r)

    for i in range(max_iter):
        if scale == 0.0:
            warnings.warn(
                'Estimated scale is 0.0 indicating that the most last '
                'iteration produced a perfect fit of the weighted data.'
            )
            break

        params = weighted_least_squares(X, y, weights=M.weights(r / scale))

        r = residual.compute(params)
        scale = _estimate_scale(r)

        # TODO
        # if _check_convergence():
        #     break

    print("params", params)
    return params
