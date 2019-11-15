from tadataka.coordinates import xy_to_yx

from skimage.feature import plot_matches


def plot_matches(ax, image1, image2, keypoints1, keypoints2):
    plot_matches_(ax, image1, image2,
                  xy_to_yx(keypoints1), xy_to_yx(keypoints2), matches12)
