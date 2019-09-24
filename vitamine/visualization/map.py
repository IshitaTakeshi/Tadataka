from vitamine.bundle_adjustment.mask import pose_mask, point_mask
from vitamine.visualization.visualizers import plot3d
from vitamine.visualization.cameras import cameras_poly3d
from vitamine.so3 import rodrigues
from matplotlib import pyplot as plt


def plot_map(camera_omegas, camera_locations, points):
    point_mask_ = point_mask(points)
    pose_mask_ = pose_mask(camera_omegas, camera_locations)

    ax = plot3d(points[point_mask_])
    ax.add_collection3d(
        cameras_poly3d(rodrigues(camera_omegas[pose_mask_]),
                       camera_locations[pose_mask_])
    )
    plt.show()
