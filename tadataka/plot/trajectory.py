from matplotlib import pyplot as plt

from tadataka.plot.common import axis3d


def plot_trajectories(trajectories, labels):
    assert(len(trajectories) == len(labels))

    ax = axis3d()
    for trajectory, label in zip(trajectories, labels):
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                label=label)
    plt.legend()
    plt.show()
