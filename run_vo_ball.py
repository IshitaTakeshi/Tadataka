from pathlib import Path

from skimage.color import rgb2gray
from skimage.io import imread

from vitamine.camera import CameraParameters
from vitamine.camera_distortion import FOV
from vitamine.dataset.tum_rgbd import TUMDataset
from vitamine.visual_odometry.visual_odometry import VisualOdometry
from vitamine.camera_distortion import FOV

from matplotlib import pyplot as plt
from vitamine.plot.visualizers import plot3d


camera_parameters = CameraParameters(focal_length=[3104.3, 3113.34],
                                     offset=[1640, 1232])
distortion_model = FOV(-0.01)
visual_odometry = VisualOdometry(camera_parameters, distortion_model,
                                 min_active_keyframes=5)

filenames = sorted(Path("./datasets/ball/").glob("*.jpg"))
for i in range(0, len(filenames)):
    image = imread(filenames[i])
    visual_odometry.add(rgb2gray(image))
    visual_odometry.try_remove()
    points = visual_odometry.export_points()

    if len(points) == 0:
        continue

    plot3d(points)
    plt.show()
