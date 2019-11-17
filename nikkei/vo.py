from autograd import numpy as np
from skimage.feature import plot_matches
from skimage.io import imread
from pathlib import Path

from tadataka.camera import CameraParameters
from tadataka.camera_distortion import FOV
from tadataka.features import extract_features, Matcher
from tadataka.plot import plot_map
from tadataka.pose import estimate_pose_change, Pose
from tadataka.plot import plot_matches
from tadataka.visual_odometry import VisualOdometry


# saba
vo = VisualOdometry(
    CameraParameters(focal_length=[2890.16, 3326.04], offset=[1640, 1232]),
    FOV(0.01),
    max_active_keyframes=6
)


# とりあえずこれを動かしてもらう

filenames = sorted(Path("./datasets/saba/").glob("*.jpg"))
filenames = [filenames[0]] + filenames[4:]
filenames = filenames[:4]

for i, filename in enumerate(filenames):
    print("filename = {}".format(filename))
    image = imread(filename)

    viewpoint = vo.add(image)

    if viewpoint < 0:
        continue  # フレームの追加に失敗した場合

    vo.try_remove()

    if i == 0:
        continue

    points, colors = vo.export_points()
    plot_map(vo.export_poses(), points, colors)
