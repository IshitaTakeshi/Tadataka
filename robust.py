from autograd import numpy as np


def rho(x, sigma=0.1):
    """
    Geman–McClure kernel
    rho(x) = x^2 / (x^2 + sigma^2)
    """

    u = np.power(x, 2)
    v = np.power(sigma, 2)
    return u / (u + v)


def psi(x, sigma):
    v = np.power(sigma, 2)
    return 2 * x * v / (np.power(x, 2) + v)


