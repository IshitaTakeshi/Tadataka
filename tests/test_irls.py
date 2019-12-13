import numpy as np
from tadataka.irls import fit
from tests.utils import relative_error


def test_fit():
    np.random.seed(3939)

    n_samples = 50
    x = np.linspace(0, 20, n_samples)
    X = np.column_stack((x, (x - 5)**2))
    X = np.hstack((np.ones((n_samples, 1)), X))

    params_true = [5, 0.5, -1.4]
    y_true = np.dot(params_true, X.T).T
    y = y_true + np.random.normal(size=n_samples, scale=0.3)
    y[[39, 41, 43, 45, 48]] -= 5.0

    params_pred = fit(X, y)

    assert(relative_error(params_true, params_pred) < 2e-2)
