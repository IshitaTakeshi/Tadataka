from pathlib import Path

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tadataka.plot.common import axis3d
from tadataka.pose import Pose
from tadataka.vo.dvo import DVO
from tadataka.dataset.tum_rgbd import TumRgbdDataset
from tadataka.camera import CameraParameters


def set_line_3d(line, data):
    line.set_data(data[:, 0:2].T)
    line.set_3d_properties(data[:, 2])


def set_image(image_axis, image):
    image_axis.set_array(image)


fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_xlim([-2, 2])
ax1.set_ylim([-2, 2])
ax1.set_zlim([-2, 2])

ax2 = fig.add_subplot(122)

camera_parameters = CameraParameters(
    focal_length=[525.0, 525.0],
    offset=[319.5, 239.5]
)


class Drawer(object):
    def __init__(self, dataset):
        self.pose = Pose.identity()
        self.dataset = dataset
        self.trajectory = np.zeros((1, 3))
        self.line = ax1.plot([0], [0], [0])[0]
        self.image_axis = ax2.imshow(dataset[0].image)

    def update(self, i):
        frame0, frame1 = self.dataset[i+0], self.dataset[i+1]

        vo = DVO(camera_parameters,
                 rgb2gray(frame0.image), frame0.depth_map,
                 rgb2gray(frame1.image))
        dpose = vo.estimate_motion(n_coarse_to_fine=6)

        self.pose = self.pose * dpose.inv()
        self.trajectory = np.vstack((self.trajectory, self.pose.t))

        set_line_3d(self.line, self.trajectory)
        set_image(self.image_axis, frame1.image)


dataset = TumRgbdDataset(Path("datasets/rgbd_dataset_freiburg1_desk"),
                         which_freiburg=1)

drawer = Drawer(dataset)

anim = animation.FuncAnimation(fig, drawer.update, 25,
                               interval=50, blit=False)

plt.show()
