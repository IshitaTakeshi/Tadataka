from skimage.color import rgb2gray


class Observations(object):
    def __init__(self, observations, window_size):
        self.observations = observations
        self.window_size = window_size
        n_observations = observations.shape[0]
        self.end = n_observations - self.window_size

    def __getitem__(self, i):
        if i < 0 or self.end < i:
             raise IndexError("Index {} is out of bounds".format(i))
        return self.observations[i:i+self.window_size]


class BaseObserver(object):
    def request(self):
        raise NotImplementedError()

    def is_running(self):
        raise NotImplementedError()


class DummyObserver(BaseObserver):
    def __init__(self, observations):
        self.observations = observations
        self.index = 0

    def request(self):
        keypoints_ = self.observations[self.index]
        self.index += 1
        return keypoints_

    def is_running(self):
        return self.index < self.observations.shape[0]


class ImageObserver(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0

    def request(self):
        image = self.dataset[self.index].image
        self.index += 1
        return image

    def is_running(self):
        return self.index < len(self.dataset)


class GrayImageObserver(ImageObserver):
    def request(self):
        image = super().request()
        return rgb2gray(image)
