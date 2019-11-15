from pathlib import Path

from skimage.color import rgb2gray

from matplotlib import pyplot as plt

from tadataka.plot.visualizers import plot3d
from tadataka.camera import CameraParameters
from tadataka.camera_distortion import FOV
from tadataka.dataset.tum_rgbd import TUMDataset
from tadataka.observations import GrayImageObserver
from tadataka.visual_odometry.visual_odometry import VisualOdometry


camera_parameters = CameraParameters(
    focal_length=[525.0, 525.0],
    offset=[319.5, 239.5]
)
dataset = TUMDataset(Path("datasets", "TUM", "rgbd_dataset_freiburg1_xyz"))

vo = VisualOdometry(camera_parameters, FOV(0.0))


for i, frame in enumerate(dataset):
    image = rgb2gray(frame.image)
    # plt.imshow(image, cmap="gray")
    vo.add(image)
    vo.try_remove()

    if vo.n_active_keyframes > 1:
        points = vo.export_points()
        plot3d(points)
        plt.show()
