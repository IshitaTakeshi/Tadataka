from autograd import numpy as np

from pathlib import Path

from skimage.color import rgb2gray

from vitamine.camera import CameraParameters
from vitamine.camera_distortion import FOV
from vitamine.dataset.tum_rgbd import TUMDataset
from vitamine.visual_odometry.visual_odometry import VisualOdometry
from vitamine.dataset.observations import (
    generate_observations, generate_translations)
from vitamine.dataset.points import cubic_lattice
from vitamine.projection import PerspectiveProjection
from vitamine.so3 import rodrigues
from vitamine.utils import random_binary, break_other_than
from vitamine.plot.visualizers import plot3d
from matplotlib import pyplot as plt


camera_parameters = CameraParameters(
    focal_length=[525.0, 525.0],
    offset=[319.5, 239.5]
)
projection = PerspectiveProjection(camera_parameters)

vo = VisualOdometry(camera_parameters, FOV(0.0))

omegas = np.random.uniform(-np.pi, np.pi, (20, 3))

points = cubic_lattice(5)
translations = generate_translations(rodrigues(omegas), points)
keypoints, _ = generate_observations(
    rodrigues(omegas), translations, points, projection)

descriptors = random_binary((len(points), 1024))
N = len(descriptors)
n_preserve = int(N * 0.5)

for keypoints_ in keypoints:
    # image = rgb2gray(frame.image)
    # plt.imshow(image, cmap="gray")
    # plt.show()

    indices = np.random.randint(0, N, n_preserve)
    vo.try_add(keypoints_, break_other_than(descriptors, indices))
    vo.try_remove()

    if vo.n_active_keyframes > 1:
        points = vo.export_points()
        plot3d(points)

    plt.show()
