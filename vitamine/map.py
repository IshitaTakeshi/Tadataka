from autograd import numpy as np

from vitamine.assertion import check_poses, check_points
from vitamine.bundle_adjustment.mask import pose_mask, point_mask


def expand_array(array, expected_size, constant=np.nan):
    n = expected_size - array.shape[0]
    assert(n >= 0)

    array = np.pad(array, ((0, n), (0, 0)),
                   mode='constant', constant_values=np.nan)

    assert(array.shape[0] == expected_size)
    return array


class Map(object):
    def __init__(self):
        self.is_initialized = False
        self.frame_index = 0

    def add(self, camera_omegas, camera_locations, points):
        if not self.is_initialized:
            self.initialize(camera_omegas, camera_locations, points)
            self.is_initialized = True
            return

        self.frame_index += 1
        self.add_poses(camera_omegas, camera_locations, self.frame_index)
        self.add_points(points)

    def initialize(self, camera_omegas, camera_locations, points):
        check_poses(camera_omegas, camera_locations)
        check_points(points)

        self.camera_omegas = camera_omegas
        self.camera_locations = camera_locations
        self.points = points

    def add_poses(self, camera_omegas, camera_locations, frame_index):
        check_poses(camera_omegas, camera_locations)

        mask = pose_mask(camera_omegas, camera_locations)
        indices = np.where(mask)[0]

        n = frame_index + camera_omegas.shape[0]
        self.camera_omegas = expand_array(self.camera_omegas, n)
        self.camera_locations = expand_array(self.camera_locations, n)

        self.camera_omegas[frame_index+indices] = camera_omegas[indices]
        self.camera_locations[frame_index+indices] = camera_locations[indices]

    def add_points(self, points):
        assert(self.points.shape[0] == points.shape[0])
        check_points(points)

        mask = point_mask(points)
        self.points[mask] = points[mask]

    def get(self):
        return self.camera_omegas, self.camera_locations, self.points
