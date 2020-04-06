from skimage.color import rgb2gray
from tadataka.camera import CameraModel


class Frame(object):
    def __init__(self, camera_model, image, pose):
        assert(image.ndim == 2)
        self.camera_model = camera_model
        self.image = image
        self.R, self.t = pose.R, pose.t
