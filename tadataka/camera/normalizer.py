
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
