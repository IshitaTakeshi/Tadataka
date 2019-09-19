class KeypointManager(object):
    def __init__(self):
        self.keypoints = []
        self.descriptors = []

    def add(self, keypoints, descriptors):
        self.keypoints.append(keypoints)
        self.descriptors.append(descriptors)

    def get(self, i, indices):
        keypoints = self.keypoints[i]
        descriptors = self.descriptors[i]
        return keypoints[indices], descriptors[indices]

    def size(self, i):
        return len(self.keypoints[i])
