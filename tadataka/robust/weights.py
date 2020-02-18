import numpy as np


def compute_weights_student_t(r, nu=5, n_iter=10):
    # Kerl Christian, Jürgen Sturm, and Daniel Cremers.
    # "Robust odometry estimation for RGB-D cameras."
    # Robotics and Automation (ICRA)

    def weights(variance):
        return (nu + 1) / (nu + s / variance)

    s = np.power(r, 2)

    variance = 1.0
    for i in range(n_iter):
        variance = np.mean(s * weights(variance))

    return np.sqrt(weights(variance))


def tukey(x, beta):
    w = np.zeros(x.shape)
    mask = np.abs(x) <= beta
    w[mask] = np.power(1 - np.power(x[mask] / beta, 2), 2)
    return w


def median_absolute_deviation(x):
    return np.median(np.abs(x - np.median(x)))


def compute_weights_tukey(r, beta=4.6851, c=1.4826):
    # Equation 4.28 in the paper
    sigma_mad = c * median_absolute_deviation(r)
    return tukey(r / sigma_mad, beta)


def compute_weights_huber(r, k=1.345):
    weights = np.ones(r.shape)
    abs_ = np.abs(r)
    mask = abs_ > k
    weights[mask] = k / abs_[mask]
    return weights
