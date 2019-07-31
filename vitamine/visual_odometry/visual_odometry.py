from autograd import numpy as np

from vitamine.bundle_adjustment.bundle_adjustment import bundle_adjustment_core
from vitamine.bundle_adjustment.initializers import (
    PoseInitializer, PointInitializer)
from vitamine.visual_odometry.local_ba import LocalBundleAdjustment
from vitamine.bundle_adjustment.mask import pose_mask, point_mask
from vitamine.bundle_adjustment.pnp import estimate_pose, estimate_poses
from vitamine.rigid.coordinates import camera_to_world
from vitamine.rigid.rotation import rodrigues
from vitamine.visual_odometry.initializers import Initializer, PointUpdater
from vitamine.assertion import check_keypoints
from vitamine.map import Map


def count_redundancy(keypoints):
    # count the number of shared points like below
    # where the boolean matrix is the keypoint mask
    #
    #             point0  point1  point2  point3
    # viewpoint0 [  True   False    True   False]
    # viewpoint1 [  True    True   False   False]
    #   n shared       2       1       1       0
    # redundancy       2   +   1   +   1   +   0  =  4

    mask = keypoint_mask(keypoints)
    return np.sum(np.all(mask, axis=0))


class VisualOdometry(object):
    def __init__(self, observations, camera_parameters):
        self.observations = observations
        self.camera_parameters = camera_parameters

    def local_ba(self, omegas, translations, points, keypoints):
        local_ba = LocalBundleAdjustment(omegas, translations, points,
                                         self.camera_parameters)
        omegas, translations, points = local_ba.compute(keypoints)
        return omegas, translations, points

    def sequence(self):
        K = self.camera_parameters.matrix
        map_ = Map()

        # keypoints.shape == (window_size, n_points, 2)
        keypoints = self.observations[0]
        initializer = Initializer(keypoints, K)
        omegas, translations, points = initializer.initialize(0, 1)

        omegas, translations, points = self.local_ba(
            omegas, translations, points, keypoints)

        map_.add(0, *camera_to_world(omegas, translations), points)

        # plot(map_)

        updater = PointUpdater(K)
        for i, keypoints in enumerate(self.observations):
            print("i =", i)

            # # shift local
            # omegas[:-1] = omegas[1:]
            # translations[:-1] = translations[1:]

            # # update the latest
            # omegas[-1], translations[-1] = estimate_pose(points, keypoints[-1], K)

            points = updater.update(points, keypoints[:-1], keypoints[-1])
            omegas, translations = estimate_poses(points, keypoints, K)
            print("omegas")
            print(omegas)

            omegas, translations, points = self.local_ba(
                omegas, translations, points, keypoints)

            map_.add(i + 1, *camera_to_world(omegas, translations), points)

            # plot(map_)


def plot(map_):
    from vitamine.visualization.visualizers import plot3d
    from vitamine.visualization.cameras import cameras_poly3d
    from matplotlib import pyplot as plt

    camera_omegas, camera_locations, global_points = map_.get()

    point_mask_ = point_mask(global_points)
    pose_mask_ = pose_mask(camera_omegas, camera_locations)

    ax = plot3d(global_points[point_mask_])
    ax.add_collection3d(
        cameras_poly3d(rodrigues(camera_omegas[pose_mask_]),
                       camera_locations[pose_mask_])
    )
    plt.show()
