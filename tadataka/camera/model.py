class CameraModel(object):
    def __init__(self, camera_parameters, distortion_model):
        self.normalizer = Normalizer(camera_parameters)
        self.camera_parameters = camera_parameters
        self.distortion_model = distortion_model

    def undistort(self, keypoints):
        return self.distortion_model.undistort(
            self.normalizer.normalize(keypoints)
        )

    def distort(self, normalized_keypoints):
        return self.normalizer.inverse(
            self.distortion_model.undistort(normalized_keypoints)
        )

    def __str__(self):
        distortion_type = type(self.distortion_model).__name__
        return ' '.join([
            distortion_type,
            str(self.camera_parameters),
            str(self.distortion_model)
        ])
