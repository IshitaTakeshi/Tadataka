import numpy as np
from tadataka.decorator import allow_1d
from tadataka.interpolation import interpolation
from tadataka.coordinates import image_coordinates
from tadataka.utils import is_in_image_range


def _normalize(_xs_map_0, _xs_map_1, us):
    xs = np.empty(us.shape)
    xs[:, 0] = interpolation(_xs_map_0, us)
    xs[:, 1] = interpolation(_xs_map_1, us)
    return xs


class NoramlizationMapTable(object):
    def __init__(self, camera_model, image_shape):
        self.image_shape = image_shape
        us = image_coordinates(image_shape)
        xs = camera_model.normalize(us)

        self._xs_map_0 = xs[:, 0].reshape(image_shape)
        self._xs_map_1 = xs[:, 1].reshape(image_shape)

    @allow_1d(which_argument=1)
    def normalize(self, us):
        assert(is_in_image_range(us, self.image_shape).all())
        return _normalize(self._xs_map_0, self._xs_map_1, us)
