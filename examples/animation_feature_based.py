from pathlib import Path

import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from tadataka.camera.io import load
from tadataka.plot import plot_map
from tadataka.plot.cameras import cameras_poly3d
from tadataka.plot.visualizers import set_aspect_equal
from tadataka.vo import FeatureBasedVO


def set_line_3d(line, data):
    line.set_data(data[:, 0:2].T)
    line.set_3d_properties(data[:, 2])


def set_points_3d(scatter, points, colors):
    scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
    scatter._facecolor3d = colors # (colors[:, 0], colors[:, 1], colors[:, 2])
    scatter._edgecolor3d = colors # (colors[:, 0], colors[:, 1], colors[:, 2])


def set_image(image_axis, image):
    image_axis.set_array(image)



def set_ax_range(ax, points, trajectory):
    min1, max1 = np.min(points, axis=0), np.max(points, axis=0)
    min2, max2 = np.min(trajectory, axis=0), np.max(trajectory, axis=0)
    min_ = np.min(np.vstack((min1, min2)), axis=0)
    max_ = np.max(np.vstack((max1, max2)), axis=0)
    ax.set_xlim([min_[0], max_[0]])
    ax.set_ylim([min_[1], max_[1]])
    ax.set_zlim([min_[2], max_[2]])
    set_aspect_equal(ax)


class Drawer(object):
    def __init__(self, fig, vo):
        self.vo = vo
        self.ax1 = fig.add_subplot(121, projection='3d')
        self.ax2 = fig.add_subplot(122)

        self.line = self.ax1.plot([0], [0], [0])[0]
        self.points = self.ax1.scatter([0], [0], [0], s=0.1)
        self.ax1.view_init(30, 30)

        image = imread(filenames[0])
        self.vo.add(image)
        self.image_axis = self.ax2.imshow(image)

    def update(self, i):
        if i == 0:
            return

        image = imread(filenames[i])
        viewpoint = vo.add(image)

        if viewpoint < 0:
            return

        vo.try_remove()

        print("Added {}".format(i))

        points, colors = vo.export_points()
        set_points_3d(self.points, points, colors)

        poses = vo.export_poses()
        trajectory = np.array([p.local_to_world().t for p in poses])

        set_line_3d(self.line, trajectory)
        set_ax_range(self.ax1, points, trajectory)
        set_image(self.image_axis, image)


camera_models = load("./datasets/nikkei/cameras.txt")
camera_model = camera_models[1]
filenames = sorted(Path("datasets/nikkei/images").glob("*.jpg"))[:240]

fig = plt.figure(figsize=(16, 10))
vo = FeatureBasedVO(camera_model, window_size=4)

drawer = Drawer(fig, vo)
anim = animation.FuncAnimation(fig, drawer.update, len(filenames),
                               interval=100, blit=False)
anim.save("feature-based-vo-saba.mp4")
# plt.show()
