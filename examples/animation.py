from pathlib import Path

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
from tadataka.plot.visualizers import set_aspect_equal


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
    set_aspect_equal(ax)


class Drawer(object):
    def __init__(self, fig, vo, dataset):
        self.ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        self.ax2 = fig.add_subplot(2, 2, 2)
        self.ax3 = fig.add_subplot(2, 2, 4)

        self.vo = vo
        self.dataset = dataset
        self.trajectory_pred = np.empty((0, 3))
        self.trajectory_true = np.empty((0, 3))
        self.line = self.ax1.plot([0], [0], [0], color='blue')[0]
        self.depth_axis = self.ax2.imshow(dataset[0].depth_map, cmap="gray")
        self.image_axis = self.ax3.imshow(dataset[0].image)

    def update(self, i):
        frame = self.dataset[i]
        pose = self.vo.estimate(frame)

        self.trajectory_pred = np.vstack((self.trajectory_pred, pose.t))
        self.trajectory_true = np.vstack((self.trajectory_true, frame.pose.t))

        set_line_3d(self.line, self.trajectory_pred)
        set_ax_range(self.ax1, self.trajectory_pred)
        set_image(self.depth_axis, frame.depth_map)
        set_image(self.image_axis, frame.image)


class TrajectoryVisualizer(object):
    def __init__(self, fig, trajetory_true, trajectory_pred):
        self.fig = fig
        self.ax = fig.add_subplot(111, projection='3d')

        P, Q = trajetory_true, trajectory_pred
        self.ax.plot(P[:, 0], P[:, 1], P[:, 2],
                     color='red', label="ground truth")
        self.ax.plot(Q[:, 0], Q[:, 1], Q[:, 2],
                     color='blue', label="prediction")
        self.ax.legend()

    def update(self, angle):
        self.ax.view_init(30, angle)
        return self.fig,


def align_trajectories(trajectory1, trajectory2):
    assert(len(trajectory1) == len(trajectory2))
    R, t, s = LeastSquaresRigidMotion(trajectory1, trajectory2).solve()
    return Transform(R, t, s)(trajectory1)


dataset = TumRgbdDataset(Path("datasets/rgbd_dataset_freiburg1_desk"),
                         which_freiburg=1)

fig = plt.figure(figsize=(16, 10))

vo = DVO()
drawer = Drawer(fig, vo, dataset)

anim = animation.FuncAnimation(fig, drawer.update, len(dataset)-1,
                               interval=50, blit=False)
anim.save("dvo-freiburg1-desk.mp4", dpi=400)
# plt.show()


fig = plt.figure(figsize=(6, 6))
visualizer = TrajectoryVisualizer(
    fig,
    drawer.trajectory_true,
    align_trajectories(drawer.trajectory_pred, drawer.trajectory_true)
)
anim = animation.FuncAnimation(fig, visualizer.update, frames=360,
                               interval=50, blit=False)
anim.save("dvo-freiburg1-desk-trajetory.mp4", dpi=400)
