from autograd import numpy as np



class Points(object):
    def __init__(self):
        self.points = np.empty((0, 3), dtype=np.float64)

    def __len__(self):
        return self.points.shape[0]

    def add(self, points):
        start = len(self.points)
        self.points = np.vstack((self.points, points))
        end = len(self.points)
        return np.arange(start, end)  # return indices of points

    def get(self, indices=None):
        if indices is None:
            return self.points
        return self.points[indices]
