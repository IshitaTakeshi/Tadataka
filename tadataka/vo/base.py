from tadataka.camera import CameraModel


class BaseVO(object):
    def __init__(self, camera_model):
        self.camera_model = camera_model
