import re

from tadataka.camera.normalizer import Normalizer
from tadataka.camera.parameters import CameraParameters
from tadataka.camera.distortion import FOV, RadTan, NoDistortion
from tadataka.decorator import allow_1d


def parse_(string):
    # split by arbitrary number of whitespaces
    params = re.split(r"\s+", string)
    distortion_type = params[0]
    params = [float(v) for v in params[1:]]
    camera_parameters = CameraParameters.from_params(params[0:4])

    dist_params = params[4:]
    if distortion_type == "FOV":
        distortion_model = FOV.from_params(dist_params)
    elif distortion_type == "RadTan":
        distortion_model = RadTan.from_params(dist_params)
    else:
        ValueError("Unknown distortion model: " + distortion_type)
    return CameraModel(camera_parameters, distortion_model)


class CameraModel(object):
    def __init__(self, camera_parameters, distortion_model):
        self.normalizer = Normalizer(camera_parameters)
        self.camera_parameters = camera_parameters

        self.distortion_model = distortion_model
        if distortion_model is None:
            self.distortion_model = NoDistortion()

    @allow_1d(which_argument=1)
    def normalize(self, keypoints):
        """
        Move keypoints from image coordinate system
        to the normalized image plane
        """
        return self.distortion_model.undistort(
            self.normalizer.normalize(keypoints)
        )

    @allow_1d(which_argument=1)
    def unnormalize(self, normalized_keypoints):
        """
        Move coordinates from the normalized image plane
        to the image coordinate system
        """
        return self.normalizer.unnormalize(
            self.distortion_model.distort(normalized_keypoints)
        )

    def __str__(self):
        distortion_type = type(self.distortion_model).__name__
        params = self.camera_parameters.params + self.distortion_model.params
        return ' '.join([distortion_type] + [str(v) for v in params])

    @staticmethod
    def fromstring(string):
        return parse_(string)

    def __eq__(self, another):
        return (self.camera_parameters == another.camera_parameters and
                self.distortion_model == another.distortion_model)


def resize(cm, scale):
    return CameraModel(
        CameraParameters(cm.camera_parameters.focal_length * scale,
                         cm.camera_parameters.offset * scale),
        cm.distortion_model
    )
