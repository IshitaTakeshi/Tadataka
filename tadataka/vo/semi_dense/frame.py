from skimage.color import rgb2gray
from tadataka.camera import CameraModel


class Frame(object):
    def __init__(self, frame):
        self.camera_model = CameraModel(
            frame.camera_model.camera_parameters,
            distortion_model=None
        )
        self.image = rgb2gray(frame.image)
        self.R, self.t = frame.pose.R, frame.pose.t
