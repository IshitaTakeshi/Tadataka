import numpy as np
import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
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


def plot_with_bar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def plot_prior(image, depth_map_true,
               depth_map_pred, variance_map_pred,
               image_cmap="gray", depth_cmap="RdBu"):

    fig = plt.figure()

    fig.suptitle("Prior")

    vmin = min(np.min(depth_map_true), np.min(depth_map_pred))
    vmax = max(np.max(depth_map_true), np.max(depth_map_pred))

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = ScalarMappable(norm=norm, cmap=depth_cmap)

    ax = fig.add_subplot(221)
    ax.set_title("frame")
    ax.imshow(image, cmap=image_cmap)

    ax = fig.add_subplot(222)
    ax.set_title("ground truth depth map")
    im = ax.imshow(depth_map_true, norm=norm, cmap=depth_cmap)

    ax = fig.add_subplot(223)
    ax.set_title("prior depth map")
    im = ax.imshow(depth_map_pred, norm=norm, cmap=depth_cmap)

    plot_with_bar(ax, im)

    ax = fig.add_subplot(224)
    ax.set_title("prior variance map")
    im = ax.imshow(variance_map_pred, cmap=depth_cmap)

    plot_with_bar(ax, im)

    plt.show()


def plot_depth(image_key, image_ref, flag_map,
               depth_map_true, depth_map_pred, variance_map,
               image_cmap="gray", depth_cmap='RdBu'):


    fig = plt.figure()

    ax = fig.add_subplot(241)
    ax.set_title("keyframe")
    ax.imshow(image_key, cmap=image_cmap)

    ax = fig.add_subplot(242)
    ax.set_title("reference frame")
    ax.imshow(image_ref, cmap=image_cmap)

    ax = fig.add_subplot(243)
    ax.set_title("flag map")
    ax.imshow(flag_to_color_map(flag_map))
    patches = [Patch(facecolor=flag_to_color(f), label=f.name) for f in FLAG]
    ax.legend(handles=patches, loc='lower left', bbox_to_anchor=(0.6, 1.05))

    mask = flag_map==FLAG.SUCCESS
    depths_pred = depth_map_pred[mask]
    depths_true = depth_map_true[mask]
    depths_diff = np.abs(depths_pred - depths_true)

    us = image_coordinates(depth_map_pred.shape)[mask.flatten()]

    vmin = min(np.min(depth_map_true), np.min(depths_pred))
    vmax = max(np.max(depth_map_true), np.max(depths_pred))
    norm = Normalize(vmin=vmin, vmax=vmax)
    mapper = ScalarMappable(norm=norm, cmap=depth_cmap)

    ax = fig.add_subplot(245)
    ax.set_title("ground truth depth")
    im = ax.imshow(depth_map_true, norm=norm, cmap=depth_cmap)
    plot_with_bar(ax, im)

    height, width = image_key.shape[0:2]

    ax = fig.add_subplot(246)
    ax.set_title("predicted depth map")
    im = ax.imshow(image_key, cmap=image_cmap)
    ax.scatter(us[:, 0], us[:, 1], s=0.5, c=mapper.to_rgba(depths_pred))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    # plot_with_bar(ax, im)

    ax = fig.add_subplot(247)
    ax.set_title("error = abs(pred - true)")
    im = ax.imshow(image_key, cmap=image_cmap)
    ax.scatter(us[:, 0], us[:, 1], s=0.5, c=mapper.to_rgba(depths_diff))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    # plot_with_bar(ax, im)

    ax = fig.add_subplot(248)
    ax.set_title("variance map")
    im = ax.imshow(variance_map)
    plot_with_bar(ax, im)

    plt.show()


def plot_warp(warp2d, gray_image0, depth_map0, gray_image1):
    from tadataka.interpolation import interpolation
    from tadataka.coordinates import image_coordinates
    from tadataka.utils import is_in_image_range
    from matplotlib import pyplot as plt

    us0 = image_coordinates(depth_map0.shape)
    depths0 = depth_map0.flatten()
    us1, depths1 = warp2d(us0, depths0)
    mask = is_in_image_range(us1, depth_map0.shape)

    fig = plt.figure()

    ax = fig.add_subplot(221)
    ax.set_title("t0 intensities")
    ax.imshow(gray_image0, cmap="gray")

    ax = fig.add_subplot(223)
    ax.set_title("t0 depth")
    ax.imshow(depth_map0, cmap="gray")

    ax = fig.add_subplot(222)
    ax.set_title("t1 intensities")
    ax.imshow(gray_image1, cmap="gray")

    ax = fig.add_subplot(224)
    ax.set_title("predicted t1 intensities")
    height, width = gray_image1.shape
    ax.scatter(us1[mask, 0], us1[mask, 1],
               c=gray_image0[us0[mask, 1], us0[mask, 0]],
               s=0.5,
               cmap="gray")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')

    plt.show()


