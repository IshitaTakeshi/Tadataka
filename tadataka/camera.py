import numpy as np


class CameraParameters(object):
    def __init__(self, focal_length, offset, skew=0.):
        assert(len(focal_length) == 2)
        assert(len(offset) == 2)

        self.focal_length = focal_length
        self.offset = offset
        self.skew = skew

    @property
    def matrix(self):
        ox, oy = self.offset
        fx, fy = self.focal_length
        s = self.skew

        return np.array([
            [fx, s, ox],
            [0., fy, oy],
            [0., 0., 1.]
        ])


class Normalizer(object):
    def __init__(self, camera_parameters):
        self.focal_length = camera_parameters.focal_length
        self.offset = camera_parameters.offset

    def normalize(self, keypoints):
        """
        Transform keypoints to the normalized plane
        (x - cx) / fx = X / Z
        (y - cy) / fy = Y / Z
        """
        return (keypoints - self.offset) / self.focal_length

    def inverse(self, normalized_keypoints):
        """
        Inverse transformation from the normalized plane
        x = fx * X / Z + cx
        y = fy * Y / Z + cy
        """
        return normalized_keypoints * self.focal_length + self.offset


class CameraModel(object):
    def __init__(self, camera_parameters, distortion_model):
        self.normalizer = Normalizer(camera_parameters)
        self.distortion_model = distortion_model

    def undistort(self, keypoints):
        return self.distortion_model.undistort(
            self.normalizer.normalize(keypoints)
        )

    def distort(self, normalized_keypoints):
        return self.normalizer.inverse(
            self.distortion_model.undistort(normalized_keypoints)
        )
