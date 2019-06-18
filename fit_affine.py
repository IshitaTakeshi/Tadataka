import skimage
from skimage import data
from skimage import transform as tf
from skimage.color import rgb2gray

# required to import numpy from autograd
# so that autograd available for all numpy functions
from autograd import numpy as np
from autograd import jacobian

from optimization.robustifiers import GemanMcClureRobustifier, SquaredRobustifier
from optimization.updaters import GaussNewtonUpdater
from optimization.optimizers import BaseOptimizer
from optimization.residuals import Residual
from optimization.errors import BaseError
from flow_estimation.keypoints import extract_keypoints
from flow_estimation.extrema_tracker import ExtremaTracker
from utils import to_2d, from_2d, is_in_image_range
from matrix import affine_trasformation, homogeneous_matrix


# we handle point coordinates P in a format:
# P[:, 0] contains x coordinates
# P[:, 1] contains y coordinates


class Error(BaseError):
    def __init__(self, residual, robustifier):
        self.residual = residual
        self.robustifier = robustifier

    def compute(self, theta):
        r = self.residual.residuals(theta)
        norms = np.linalg.norm(r.reshape(-1, 2), axis=1)
        return np.sum(self.robustifier.robustify(norms))


def initialize_theta(initial_A=None, initial_b=None):
    if initial_A is None:
        initial_A = np.random.uniform(-1, 1, size=(2, 2))

    if initial_b is None:
        initial_b = np.random.uniform(-1, 1, size=2)

    return np.concatenate((
        initial_A.flatten(),
        initial_b.flatten()
    ))


def theta_to_affine_params(theta):
    A = np.reshape(theta[0:4], (2, 2))
    b = theta[4:6]
    return A, b


def transform_image(image, theta):
    A, b = theta_to_affine_params(theta)
    W = homogeneous_matrix(A, b)
    # Note that tf.warp requires the inverse transformation
    return tf.warp(image, tf.AffineTransform(matrix=np.linalg.inv(W)))


def transformer(x, theta):
    """
    X : image coordinates of shape (n_points, 2)
    A : 2x2 transformation matrix
    b : bias term of shape (2,)
    """

    X = to_2d(x)
    A, b = theta_to_affine_params(theta)
    Y = affine_trasformation(X, A, b)
    return from_2d(Y)


def predict(keypoints1, keypoints2, initial_theta):
    residual = Residual(from_2d(keypoints1), from_2d(keypoints2), transformer)
    # TODO Geman-McClure is used in the original paper
    robustifier = SquaredRobustifier()
    updater = GaussNewtonUpdater(residual, robustifier)
    optimizer = BaseOptimizer(updater, Error(residual, robustifier))
    return optimizer.optimize(initial_theta, n_max_iter=1000)


def yx_to_xy(coordinates):
    return coordinates[:, [1, 0]]


def xy_to_yx(coordinates):
    # this is identical to 'yx_to_xy' but I prefer to name expilictly
    return yx_to_xy(coordinates)


def estimate_affine_transformation(image1, image2):
    """
    Esitmate the affine transformation from image1 to image2
    """

    keypoints1, keypoints2, matches12 = extract_keypoints(image1, image2)

    keypoints1 = yx_to_xy(keypoints1[matches12[:, 0]])
    keypoints2 = yx_to_xy(keypoints2[matches12[:, 1]])

    initial_theta = initialize_theta()
    theta_pred = predict(keypoints1, keypoints2, initial_theta)
    A, b = theta_to_affine_params(theta_pred)
    return keypoints1, keypoints2, A, b


def plot_keypoints(ax, image, keypoints, **kwargs):
    print(skimage.img_as_float(image))
    ax.imshow(skimage.img_as_float(image), cmap='gray')
    ax.scatter(keypoints[:, 0], keypoints[:, 1], **kwargs)


def round_to_int(X):
    return np.round(X, 0).astype(np.int64)


def plot(ax, image, keypoints, lambda_):
    size = 2

    ax.set_title(r"$\lambda = {}$".format(lambda_))

    plot_keypoints(ax, image, keypoints,
                   c="yellow", s=size, label=r"$\overline{\mathbf{x}}$")

    keypoints = ExtremaTracker(image, keypoints, lambda_).optimize()

    plot_keypoints(ax, image, keypoints,
                   c="red", s=size, label=r"$\arg \max \, F(\mathbf{x})$")


def test_extrema_tracker(image, keypoints):
    lambdas = [0, 1e-4, 1e-2, 1, 1e2, 1e4]
    fig, axes = plt.subplots(nrows=2, ncols=len(lambdas)//2)
    axes = axes.flatten()

    for ax, lambda_ in zip(axes, lambdas):
        ax.axis('off')
        plot(ax, image, keypoints, lambda_)

    axes[-1].legend(loc="best", borderaxespad=0.1)

    plt.show()


if __name__ == "__main__":
    from scipy.optimize import least_squares
    from matplotlib import pyplot as plt

    np.set_printoptions(precision=5, suppress=True, linewidth=100,
                        formatter={'float': '{: 08.5f}'.format})

    image = rgb2gray(data.astronaut())

    theta_true = np.array([1.0, 0.2, -0.2, 1.0, -100.0, 20.0])
    print("ground truth                     : ", theta_true)

    image_transformed = transform_image(image, theta_true)

    keypoints, keypoints_transformed, A, b =\
        estimate_affine_transformation(image, image_transformed)

    keypoints_pred = affine_trasformation(keypoints, A, b)
    keypoints_pred = round_to_int(keypoints_pred)

    mask = is_in_image_range(keypoints_pred, image.shape[0:2])
    keypoints_pred = keypoints_pred[mask]

    test_extrema_tracker(image, keypoints)
