from skimage import data
from skimage import transform as tf
from skimage.feature import plot_matches
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# required to import numpy from autograd
# so that autograd available for all numpy functions
from autograd import numpy as np
from autograd import jacobian

from robustifiers import GemanMcClureRobustifier, SquaredRobustifier
from optimizers import Optimizer
from keypoints import extract_keypoints


def affine_parameters_from_theta(theta):
    A = np.reshape(theta[0:4], (2, 2))
    b = theta[4:6]
    return A, b


def affine_transform(X, A, b):
    """
    X : image coordinates of shape (n_points, 2)
    A : 2x2 transformation matrix
    b : bias term of shape (2,)
    """
    return np.dot(A, X.T).T + b


def gauss_newton_update(J, r, weights=None):
    # Not exactly the same as the equation of Gauss-Newton update
    # d = inv (J^T * J) * J * r
    # however, it works better than implementing the equation malually
    theta, error, _, _ = np.linalg.lstsq(J, r, rcond=None)
    return theta


class GaussNewtonUpdater(object):
    def __init__(self, residual, robustifier):
        self.residual = residual
        self.robustifier = robustifier

    def jacobian(self, theta):
        return jacobian(self.residuals)(theta)

    def residuals(self, theta):
        r = self.residual.residuals(theta)
        return r.flatten()

    def compute(self, theta):
        r = self.residuals(theta)
        J = self.jacobian(theta)
        # weights = self.robustifier.weights(r)
        return gauss_newton_update(J, r)


class Error(object):
    def __init__(self, residual, robustifier=None):
        self.residual = residual
        self.robustifier = robustifier

        if robustifier is None:
            self.robustifier = SquaredRobustifier()

    def compute(self, theta):
        R = self.residual.residuals(theta)
        norms = np.linalg.norm(R, axis=1)
        return np.sum(self.robustifier.robustify(norms))


class Residual(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def residuals(self, theta):
        """
        Returns:
            residuals of shape (n_points, 2)
        """
        # R[i, :] row has d_i = y_i - (A * x_i + b)
        A, b = affine_parameters_from_theta(theta)
        return self.Y - affine_transform(self.X, A, b)


def initialize_theta(initial_A=None, initial_b=None):
    if initial_A is None:
        initial_A = np.random.uniform(-1, 1, size=(2, 2))

    if initial_b is None:
        initial_b = np.random.uniform(-1, 1, size=2)

    return np.concatenate((
        initial_A.flatten(),
        initial_b.flatten()
    ))



def get_affine_transformation(A, b):
    W = np.identity(3)
    W[0:2, 0:2] = A
    W[0:2, 2] = b
    return tf.AffineTransform(matrix=W)


def transform_image(image, theta):
    A, b = affine_parameters_from_theta(theta)
    return tf.warp(image, get_affine_transformation(A, b))


def predict(residual, initial_theta):
    robustifier = SquaredRobustifier()
    updater = GaussNewtonUpdater(residual, robustifier)
    optimizer = Optimizer(updater, Error(residual))
    return optimizer.optimize(initial_theta, n_max_iter=1000)


if __name__ == "__main__":
    from scipy.optimize import least_squares

    np.set_printoptions(precision=5, suppress=True,
                        formatter={'float': '{: 0.5f}'.format})

    fig, ax = plt.subplots(nrows=3, ncols=1)
    plt.gray()

    image = rgb2gray(data.astronaut())

    theta_true = np.array([1.0, 0.0, 0.0, 1.0, -100.0, 0.0])

    image_true = transform_image(image, theta_true)
    keypoints1, keypoints2, matches12 = extract_keypoints(image, image_true)

    print("ground truth                     : ", theta_true)

    plot_matches(ax[0], image, image_true, keypoints1, keypoints2, matches12)
    ax[0].axis('off')
    ax[0].set_title("Original Image vs. Transformed Image")

    initial_theta = initialize_theta()
    residual = Residual(
        keypoints1[matches12[:, 0]],
        keypoints2[matches12[:, 1]]
    )
    theta_pred = predict(residual, initial_theta)
    image_pred = transform_image(image, theta_pred)
    print("predicted by gauss-newton        : ", theta_pred)

    plot_matches(ax[1], image, image_pred, keypoints1, keypoints2, matches12)
    ax[1].axis('off')
    ax[1].set_title("Original Image vs. "
                    "Estimated by Gauss-Newton")


    res = least_squares(Error(residual).compute, initial_theta)
    image_pred = transform_image(image, res.x)
    print("predicted by scipy least squares : ", res.x)

    plot_matches(ax[2], image, image_pred, keypoints1, keypoints2, matches12)

    ax[2].axis('off')
    ax[2].set_title("Original Image vs. "
                    "Estimated by SciPy least squares")
    plt.show()
