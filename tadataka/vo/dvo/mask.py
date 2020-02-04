import numpy as np

# from tadataka.utils import is_in_image_range
#
#
# def compute_mask(depth_map, pixel_coordinates):
#     # depth_map and pixel_coordinates has to be in the same coordinate system
#
#     depth_mask = depth_map > 0
#     range_mask = is_in_image_range(pixel_coordinates, depth_map.shape)
#     range_mask = range_mask.reshape(depth_map.shape)
#     return np.logical_and(depth_mask, range_mask)

import numpy as np


def is_in_range(image_shape, coordinates):
    height, width = image_shape[:2]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    mask_x = np.logical_and(0 <= xs, xs <= width-1)
    mask_y = np.logical_and(0 <= ys, ys <= height-1)
    return np.logical_and(mask_x, mask_y)


def compute_mask(depth_map, pixel_coordinates):
    # depth_map and pixel_coordinates has to be in the same coordinate system

    depth_mask = depth_map > 0
    range_mask = is_in_range(depth_map.shape, pixel_coordinates)
    range_mask = range_mask.reshape(depth_map.shape)
    return np.logical_and(depth_mask, range_mask)
