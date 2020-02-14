
class Normalizer(object):
    def __init__(self, camera_parameters):
        self.focal_length = camera_parameters.focal_length
        self.offset = camera_parameters.offset

    def normalize(self, keypoints):
        """
        Transform keypoints to the normalized plane
        (x - cx) / fx = X / Z
        (y - cy) / fy = Y / Z

        Returns:
            Normalized keypoints of the shape (n_keypoints, 2)
        """
        return (keypoints - self.offset) / self.focal_length

    def unnormalize(self, normalized_keypoints):
        """
        Inverse transformation from the normalized plane
        x = fx * (X / Z) + cx
        y = fy * (Y / Z) + cy

        Returns:
            Keypoints of the shape (n_keypoints, 2)
        """
        return normalized_keypoints * self.focal_length + self.offset
