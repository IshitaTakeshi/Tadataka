from tadataka.camera import CameraModel


class BaseVO(object):
    def __init__(self, camera_parameters, distortion_model):
        self.camera_model = CameraModel(camera_parameters, distortion_model)
