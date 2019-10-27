from pathlib import Path

from skimage.color import rgb2gray

from matplotlib import pyplot as plt

from vitamine.plot.visualizers import plot3d
from vitamine.camera import CameraParameters
from vitamine.camera_distortion import FOV
from vitamine.dataset.tum_rgbd import TUMDataset
from vitamine.observations import GrayImageObserver
from vitamine.visual_odometry.visual_odometry import VisualOdometry


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
