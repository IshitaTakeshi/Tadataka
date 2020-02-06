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
from tadataka.rigid_motion import LeastSquaresRigidMotion
from tadataka.rigid_transform import Transform


def set_line_3d(line, data):
    line.set_data(data[:, 0:2].T)
    line.set_3d_properties(data[:, 2])


def set_image(image_axis, image):
    image_axis.set_array(image)


camera_parameters = CameraParameters(
    focal_length=[525.0, 525.0],
    offset=[319.5, 239.5]
)


def set_ax_range(ax, data):
    min_ = np.min(data, axis=0)
    max_ = np.max(data, axis=0)
    ax.set_xlim([min_[0], max_[0]])
    ax.set_ylim([min_[1], max_[1]])
    ax.set_zlim([min_[2], max_[2]])


class Drawer(object):
    def __init__(self, fig, dataset):
        self.ax1 = fig.add_subplot(121, projection='3d')
        self.ax2 = fig.add_subplot(122)

        self.pose = Pose.identity()
        self.dataset = dataset
        self.trajectory_pred = self.pose.t
        self.trajectory_true = self.dataset[0].pose.t
        self.line = self.ax1.plot([0], [0], [0])[0]
        self.image_axis = self.ax2.imshow(dataset[0].image)

    def update(self, i):
        frame0, frame1 = self.dataset[i+0], self.dataset[i+1]

        vo = DVO(camera_parameters,
                 rgb2gray(frame0.image), frame0.depth_map,
                 rgb2gray(frame1.image))
        dpose = vo.estimate_motion(n_coarse_to_fine=6)

        self.pose = self.pose * dpose.inv()
        self.trajectory_pred = np.vstack((self.trajectory_pred, self.pose.t))
        self.trajectory_true = np.vstack((self.trajectory_true, frame1.pose.t))

        set_line_3d(self.line, self.trajectory_pred)
        set_ax_range(self.ax1, self.trajectory_pred)
        set_image(self.image_axis, frame1.image)



dataset = TumRgbdDataset(Path("datasets/rgbd_dataset_freiburg1_desk"),
                         which_freiburg=1)

fig = plt.figure(figsize=(16, 10))
drawer = Drawer(fig, dataset)

anim = animation.FuncAnimation(fig, drawer.update, len(dataset)-1,
                               interval=50, blit=False)
anim.save("dvo-freiburg1-desk.mp4")
# plt.show()

m = LeastSquaresRigidMotion(drawer.trajectory_pred, drawer.trajectory_true)
R, t, s = m.solve()
Q = drawer.trajectory_true
P = Transform(R, t, s)(drawer.trajectory_pred)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(Q[:, 0], Q[:, 1], Q[:, 2], label="true")
ax.plot(P[:, 0], P[:, 1], P[:, 2], label="pred")

plt.legend()
plt.show()

