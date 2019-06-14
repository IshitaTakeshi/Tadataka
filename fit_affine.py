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
from updaters import GaussNewtonUpdater
from optimizers import Optimizer
from residuals import Residual
from errors import Error
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


# there should be a better name?
class SumRobustifiedError(Error):
    def __init__(self, residual, robustifier):
        self.residual = residual
        self.robustifier = robustifier

    def compute(self, theta):
        r = self.residual.residuals(theta)
        norms = np.linalg.norm(r.reshape(-1, 2), axis=1)
        return np.sum(self.robustifier.robustify(norms))


def from_2d(x):
    return x.flatten()


def to_2d(x):
    return x.reshape(-1, 2)


def transformer(x, theta):
    """
    X : image coordinates of shape (n_points, 2)
    A : 2x2 transformation matrix
    b : bias term of shape (2,)
    """

    X = to_2d(x)
    A, b = affine_parameters_from_theta(theta)
    y = affine_transform(X, A, b)
    return from_2d(y)


def initialize_theta(initial_A=None, initial_b=None):
    if initial_A is None:
        initial_A = np.random.uniform(-1, 1, size=(2, 2))

    if initial_b is None:
        initial_b = np.random.uniform(-1, 1, size=2)

    return np.concatenate((
        initial_A.flatten(),
        initial_b.flatten()
    ))


def get_affine_matrix(A, b):
    W = np.identity(3)
    W[0:2, 0:2] = A
    W[0:2, 2] = b
    return W


def transform_image(image, theta):
    A, b = affine_parameters_from_theta(theta)
    W = get_affine_matrix(A, b)
    # Note that tf.warp requires the inverse transformation
    return tf.warp(image, tf.AffineTransform(matrix=np.linalg.inv(W)))


def predict(keypoints1, keypoints2, initial_theta):
    residual = Residual(from_2d(keypoints1), from_2d(keypoints2), transformer)
    # TODO Geman-McClure is used in the original paper
    robustifier = SquaredRobustifier()
    updater = GaussNewtonUpdater(residual, robustifier)
    optimizer = Optimizer(updater, SumRobustifiedError(residual, robustifier))
    return optimizer.optimize(initial_theta, n_max_iter=1000)


def plot(image_original, image_true, image_pred,
         keypoints1, keypoints2, matches12):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    plt.gray()

    plot_matches(ax[0], image_original, image_true,
                 keypoints1, keypoints2, matches12)
    ax[0].axis('off')
    ax[0].set_title("Original Image vs. Transformed Image")

    plot_matches(ax[1], image_original, image_pred,
                 keypoints1, keypoints2, matches12)
    ax[1].axis('off')
    ax[1].set_title("Original Image vs. Predicted Image")

    plt.show()


def yx_to_xy(coordinates):
    return coordinates[:, [1, 0]]


def random_rotation_matrix():
    A = np.random.uniform(-1, 1, (2, 2))
    return np.linalg.svd(np.dot(A.T, A))[0]


def estimate_image_transformation():
    theta_true = np.array([1.0, 0.2, -0.2, 1.0, -100.0, 20.0])
    print("ground truth                     : ", theta_true)

    image_original = rgb2gray(data.astronaut())
    image_true = transform_image(image_original, theta_true)

    keypoints1, keypoints2, matches12 =\
        extract_keypoints(image_original, image_true)

    initial_theta = initialize_theta()
    theta_pred = predict(yx_to_xy(keypoints1[matches12[:, 0]]),
                         yx_to_xy(keypoints2[matches12[:, 1]]),
                         initial_theta)

    print("predicted by gauss newton        : ", theta_pred)
    image_pred = transform_image(image_original, theta_pred)

    plot(image_original, image_true, image_pred,
         keypoints1, keypoints2, matches12)


def test_estimator():
    X = np.array([[1, 2, 0, 3],
                  [2, 0, 1, 1]]).T

    theta_true = np.array([1.2, 0.1, -0.1, 1.2, 1.0, 2.0])
    A, b = affine_parameters_from_theta(theta_true)
    Y = affine_transform(X, A, b)

    initial_theta = initialize_theta()
    theta_pred = predict(X, Y, initial_theta)
    print(theta_true)
    print(theta_pred)


if __name__ == "__main__":
    from scipy.optimize import least_squares

    np.set_printoptions(precision=5, suppress=True,
                        formatter={'float': '{: 0.5f}'.format})

    estimate_image_transformation()
