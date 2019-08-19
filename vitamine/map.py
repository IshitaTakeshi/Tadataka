from autograd import numpy as np

from vitamine.assertion import check_poses, check_points


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
        self.camera_omegas = np.vstack((self.camera_omegas, camera_omegas))
        self.camera_locations = np.vstack((self.camera_locations, camera_locations))

    def add_points(self, points):
        check_points(points)
        self.points = np.vstack((self.points, points))

    def get(self):
        return self.camera_omegas, self.camera_locations, self.points
