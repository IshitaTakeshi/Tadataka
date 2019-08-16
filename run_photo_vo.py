from pathlib import Path

from vitamine.camera import CameraParameters
from vitamine.dataset.tum_rgbd import TUMDataset
from vitamine.observations import GrayImageObserver
from vitamine.visual_odometry.visual_odometry import VisualOdometry


camera_parameters = CameraParameters(
    focal_length=[525.0, 525.0],
    offset=[319.5, 239.5]
)
dataset = TUMDataset(Path("datasets", "TUM", "rgbd_dataset_freiburg1_desk"))

observer = GrayImageObserver(dataset)
visual_odometry = VisualOdometry(observer, camera_parameters)
visual_odometry.sequence()
