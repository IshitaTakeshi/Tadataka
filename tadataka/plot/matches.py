from matplotlib import pyplot as plt
from skimage.feature import plot_matches as plot_matches_

from tadataka.coordinates import xy_to_yx


def plot_matches(image1, image2, keypoints1, keypoints2, matches12):
    fig, ax = plt.subplots()
    plot_matches_(ax, image1, image2,
                  xy_to_yx(keypoints1), xy_to_yx(keypoints2), matches12)
    plt.show()
