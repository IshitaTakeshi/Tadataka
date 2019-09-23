from pathlib import Path

from skimage.color import rgb2gray

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

from matplotlib import pyplot as plt

for i, frame in enumerate(dataset):
    image = rgb2gray(frame.image)
    plt.imshow(image, cmap="gray")
    plt.show()
    vo.add(image)
    vo.try_remove()
    print(f"frame index = {i}")
