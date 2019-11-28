from pathlib import Path

import numpy as np
from skimage.io import imread

from tadataka.visual_odometry import VitaminE
from tadataka.camera import CameraParameters
from tadataka.camera_distortion import FOV


# saba
vo = VitaminE(
    CameraParameters(focal_length=[2890.16, 3326.04], offset=[1640, 1232]),
    FOV(0.01),
    window_size=6
)


filenames = sorted(Path("./datasets/saba/").glob("*.jpg"))
filenames = [filenames[0]] + filenames[4:]


for i, filename in enumerate(filenames[:3]):
    image = imread(filename)
    print("Adding {}-th frame".format(i))
    print("filename = {}".format(filename))

    viewpoint = vo.add(image)

    if viewpoint < 0:
        continue

    vo.try_remove()
    print("{}-th Frame Added".format(i))

    if i == 0:
        continue

# points, colors = vo.export_points()
# plot_map(vo.export_poses(), points, colors)
