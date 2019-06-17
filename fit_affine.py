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
from optimization.optimizers import Optimizer
from optimization.residuals import Residual
from optimization.errors import Error
from flow_estimation.keypoints import extract_keypoints
from curvature_extrema.image_curvature import curvature
from utils import affine_matrix, to_2d, from_2d


# we handle point coordinates P in a format:
# P[:, 0] contains x coordinates
# P[:, 1] contains y coordinates


def theta_to_affine_params(theta):
    A = np.reshape(theta[0:4], (2, 2))
    b = theta[4:6]
    return A, b


def affine_params_to_theta(A, b):
    return np.concatenate((A.flatten(), b))


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


def transformer(x, theta):
    """
    X : image coordinates of shape (n_points, 2)
    A : 2x2 transformation matrix
    b : bias term of shape (2,)
    """

    X = to_2d(x)
    A, b = theta_to_affine_params(theta)
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


def transform_image(image, theta):
    A, b = theta_to_affine_params(theta)
    W = affine_matrix(A, b)
    # Note that tf.warp requires the inverse transformation
    return tf.warp(image, tf.AffineTransform(matrix=np.linalg.inv(W)))


def predict(keypoints1, keypoints2, initial_theta):
    residual = Residual(from_2d(keypoints1), from_2d(keypoints2), transformer)
    # TODO Geman-McClure is used in the original paper
    robustifier = SquaredRobustifier()
    updater = GaussNewtonUpdater(residual, robustifier)
    optimizer = Optimizer(updater, SumRobustifiedError(residual, robustifier))
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


def search_maximum(coordinates, K, robustifier, n_max_iter=20, lambda_=0.3):
    def regularizer(x):
        return 1 - robustifier.robustify(x)

    def F(P, p0):
        R = regularizer(np.linalg.norm(P - p0, 1))
        xs, ys = P[:, 0], P[:, 1]
        return K[ys, xs] + lambda_ * R

    def diffs():
        xs, ys = np.meshgrid([-1, 0, 1], [-1, 0, 1])
        return np.vstack((xs.flatten(), ys.flatten())).T

    diffs_ = diffs()

    def get_neighbors(p):
        return p + diffs_

    def search(p0):
        p = np.copy(p0)
        for i in range(n_max_iter):
            neighbors = get_neighbors(p)
            argmax = np.argmax(F(neighbors, p0))
            p = neighbors[argmax]
        return p

    for i in range(coordinates.shape[0]):
        coordinates[i] = search(coordinates[i])
    return coordinates


def is_in_image_range(points, image_shape):
    height, width = image_shape
    xs, ys = keypoints_pred[:, 0], keypoints_pred[:, 1]
    mask_x = np.logical_and(0 <= xs, xs < width)
    mask_y = np.logical_and(0 <= ys, ys < width)
    return np.logical_and(mask_x, mask_y)


def plot_keypoints(ax, image, keypoints, **kwargs):
    print(skimage.img_as_float(image))
    ax.imshow(skimage.img_as_float(image), cmap='gray')
    ax.scatter(keypoints[:, 0], keypoints[:, 1], **kwargs)


def round_to_int(X):
    return np.round(X, 0).astype(np.int64)


def plot(ax, image_transformed, keypoints_pred, lambda_):
    size = 2

    ax.set_title(r"$\lambda = {}$".format(lambda_))

    plot_keypoints(ax, image_transformed, keypoints_pred,
                   c='red', s=size, label="predicted")

    keypoints_corrected = search_maximum(keypoints_pred, K, robustifier,
                                         lambda_=lambda_)
    plot_keypoints(ax, image_transformed, keypoints_corrected,
                   c='blue', s=size, label="corrected")


if __name__ == "__main__":
    from scipy.optimize import least_squares
    from matplotlib import pyplot as plt

    np.set_printoptions(precision=5, suppress=True,
                        formatter={'float': '{: 0.5f}'.format})

    image = rgb2gray(data.camera())

    theta_true = np.array([1.0, 0.2, -0.2, 1.0, -100.0, 20.0])
    print("ground truth                     : ", theta_true)

    image_transformed = transform_image(image, theta_true)

    keypoints, keypoints_transformed, A, b =\
        estimate_affine_transformation(image, image_transformed)

    keypoints_pred = affine_transform(keypoints, A, b)
    keypoints_pred = round_to_int(keypoints_pred)

    mask = is_in_image_range(keypoints_pred, image.shape[0:2])
    keypoints_pred = keypoints_pred[mask]

    K = curvature(image_transformed)
    robustifier = GemanMcClureRobustifier()

    fig, ax = plt.subplots(nrows=1, ncols=3)

    plot(ax[0], image_transformed, keypoints_pred,
         lambda_=1.0)
    plot(ax[1], image_transformed, keypoints_pred,
         lambda_=0.01)
    plot(ax[2], image_transformed, keypoints_pred,
         lambda_=0.0001)

    ax[2].legend(loc="best", borderaxespad=0.1)

    plt.show()
