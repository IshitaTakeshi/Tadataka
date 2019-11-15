from matplotlib import pyplot as plt
from skimage.feature import plot_matches as skimage_plot_matches
from tadataka.plot.common import axis3d
from tadataka.coordinates import xy_to_yx


def plot_masked_keypoints(X, mask, true_label, false_label):
    fig, ax = plt.subplots()

    ax.scatter(X[mask, 0], X[mask, 1],
               label=true_label, color='b', marker='.')
    ax.scatter(X[~mask, 0], X[~mask, 1],
               label=false_label, color='r', marker='.')
    ax.grid(True)
    ax.legend()

    plt.show()


def plot_masked_points(P, mask, true_label, false_label):
    ax = axis3d()
    ax.scatter(P[mask, 0], P[mask, 1], P[mask, 2], 'b.', label=true_label)
    ax.scatter(P[~mask, 0], P[~mask, 1], P[~mask, 2], 'r.', label=false_label)
    ax.legend()
    plt.show()


def plot_matches(image0, image1, keypoints0, keypoints1, matches01):
    fig, ax = plt.subplots()
    skimage_plot_matches(ax,
        image0, image1,
        xy_to_yx(keypoints0), xy_to_yx(keypoints1),
        matches01
    )
