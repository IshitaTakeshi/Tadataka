from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from vitamine.visualization.visualizers import object_color


class VisualOdometryAnimation(object):
    def __init__(self, fig, ax, frames, interval=100):
        self.ax = ax
        self.animation = FuncAnimation(fig, self.animate, frames=frames,
                                       interval=interval)

    def animate(self, args):
        omegas, translations, points = args
        print("points.shape", points.shape)
        return self.ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=object_color(points)
        )

    def plot(self):
        plt.show()
