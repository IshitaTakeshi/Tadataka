import numpy as np
import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tadataka.coordinates import image_coordinates


def flag_to_color(flag):
    if flag == FLAG.SUCCESS:
        return np.array([0, 1, 0])  # green
    if flag == FLAG.KEY_OUT_OF_RANGE:
        return np.array([0, 0, 0])  # black
    if flag == FLAG.EPIPOLAR_TOO_SHORT:
        return np.array([1, 0, 0])  # red
    if flag == FLAG.INSUFFICIENT_GRADIENT:
        return np.array([0, 0, 1])  # blue
    raise ValueError


def flag_to_color_map(flag_map):
    color_map = np.empty((*flag_map.shape, 3))
    for x, y in image_coordinates(flag_map.shape):
        color_map[y, x] = flag_to_color(flag_map[y, x])
    return color_map


class PlotWarp(object):
    def __init__(self, image_key, image_ref,
                 camera_model_key, camera_model_ref,
                 pose_key, pose_ref):
        self.image_key = image_key
        self.image_ref = image_ref
        self.warp = Warp(camera_model_key, camera_model_ref,
                         pose_key, pose_ref)

    def __call__(self, u_key, depth_key):
        u_ref = self.warp(u_key, depth_key)

        fig = plt.figure()

        ax = fig.add_subplot(121)
        ax.set_title("keyframe")
        ax.imshow(self.image_key)
        ax.scatter(u_key[0], u_key[1], c="red")

        ax = fig.add_subplot(122)
        ax.set_title("reference frame")
        ax.imshow(self.image_ref)
        ax.scatter(u_ref[0], u_ref[1], c="red")

        plt.show()


def plot_propagation(image0, image1, depth_map0, depth_map1,
                     us1, depths_pred1):
    cmap='RdBu'

    fig = plt.figure()

    ax = fig.add_subplot(231)
    ax.set_title("frame 0")
    ax.imshow(image0)

    ax = fig.add_subplot(232)
    ax.set_title("frame 1")
    ax.imshow(image1)

    vmin = min(np.min(depth_map0),
               np.min(depth_map1),
               np.min(depths_pred1))
    vmax = max(np.max(depth_map0),
               np.max(depth_map1),
               np.max(depths_pred1))

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = ScalarMappable(norm=norm, cmap=cmap)

    ax = fig.add_subplot(234)
    ax.set_title("depth map 0")
    im = ax.imshow(depth_map0, norm=norm, cmap=cmap)

    ax = fig.add_subplot(235)
    ax.set_title("predicted depth map 1")
    ax.imshow(image1)
    height, width = image1.shape[0:2]
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.scatter(us1[:, 0], us1[:, 1],
               s=1.0, c=mapper.to_rgba(depths_pred1))

    ax = fig.add_subplot(236)
    ax.set_title("ground truth depth map 1")
    im = ax.imshow(depth_map1, norm=norm, cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.show()


def plot_depth(image_key, image_ref, flag_map,
               depth_map_true, depth_map_pred, variance_map, cmap='RdBu'):
    fig = plt.figure()

    ax = fig.add_subplot(231)
    ax.set_title("keyframe")
    ax.imshow(image_key)

    ax = fig.add_subplot(232)
    ax.set_title("reference frame")
    ax.imshow(image_ref)

    ax = fig.add_subplot(233)
    ax.set_title("flag map")
    ax.imshow(flag_to_color_map(flag_map))
    patches = [Patch(facecolor=flag_to_color(f), label=f.name) for f in FLAG]
    ax.legend(handles=patches, loc='lower left', bbox_to_anchor=(0.6, 1.05))

    mask = flag_map==FLAG.SUCCESS
    depths_pred = depth_map_pred[mask]
    us = image_coordinates(depth_map_pred.shape)[mask.flatten()]

    vmin = min(np.min(depth_map_true), np.min(depths_pred))
    vmax = max(np.max(depth_map_true), np.max(depths_pred))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = ScalarMappable(norm=norm, cmap=cmap)

    ax = fig.add_subplot(234)
    ax.set_title("predicted depth map")
    ax.imshow(image_key)
    height, width = image_key.shape[0:2]
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.scatter(us[:, 0], us[:, 1],
               s=1.0, c=mapper.to_rgba(depths_pred))

    ax = fig.add_subplot(235)
    ax.set_title("ground truth depth")
    im = ax.imshow(depth_map_true, norm=norm, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = fig.add_subplot(236)
    ax.set_title("variance map")
    im = ax.imshow(variance_map)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.show()
