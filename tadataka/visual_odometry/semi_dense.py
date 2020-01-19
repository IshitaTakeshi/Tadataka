import numpy as np
import numba

from tadataka.utils import is_in_image_range
from tadataka.matrix import to_homogeneous
from tadataka.projection import pi
from tadataka.rigid_transform import transform
from tadataka.interpolate import interpolate
from tadataka.triangulation import DepthFromTriangulation
from tadataka.pose import Pose


def normalize_length(v):
    return v / np.linalg.norm(v)


def inverse_projection(camera_model, x, depth):
    return depth * to_homogeneous(x)


def calc_epipolar_direction(coordinate, t, K):
    return normalize_length(coordinate - projection(t, K))


# TODO use numba for acceleration
def convolve(a, b, error_func):
    # reverese b because the second argument is reversed in the computation
    N = len(b)
    return np.array([error_func(a[i:i+N], b) for i in range(len(a)-N+1)])


def calc_inv_depths(ref_coordinate, key_coordinate,
                   R_key_to_ref, t_key_to_ref):
    K_inv = np.linalg.inv(K)

    inv_ref_coordinate = np.dot(K_inv, ref_coordinate)
    inv_key_coordinate = np.dot(K_inv, key_coordinate)

    rot_inv_x = np.dot(R_key_to_ref, inv_key_coordinate)

    n = inv_ref_coordinate[0:2] * t_key_to_ref - t_key_to_ref[0:2]
    d = rot_inv_x[0:2] - ref_coordinate * rot_inv_x[2]
    return d / n


def calc_alphas(ref_coordinate, key_coordinate, search_step,
                R_key_to_ref, t_key_to_ref):
    K_inv = np.linalg.inv(K)

    inv_ref_coordinate = np.dot(K_inv, ref_coordinate)
    inv_key_coordinate = np.dot(K_inv, key_coordinate)

    rot_inv_x = np.dot(R_key_to_ref, inv_key_coordinate)
    n = inv_ref_coordinate[0:2] * t_key_to_ref - t_key_to_ref[0:2]
    d = rot_inv_x[0:2] * t_key_to_ref[2] - rot_inv_x[2] * t_key_to_ref[0:2]

    return search_step * K_inv[[0, 1], [0, 1]] * d / (n * n)


def coordinates_along_line(start, step, disparities):
    return start + np.outer(disparities, step)


def depth_coordinate(search_step):
    return 0 if np.abs(search_step[0]) > np.abs(search_step[1]) else 1


def calc_error(v1, v2):
    d = (v2 - v1).flatten()
    return np.dot(d, d)


def search_intensities(intensities_ref, intensities_key, error_func):
    errors = convolve(intensities_ref, intensities_key, error_func)
    assert(len(errors) == len(intensities_ref) - len(intensities_key) + 1)
    return np.argmin(errors) + (len(intensities_key) - 1) // 2


def photometric_disparity_error(epipolar_direction):
    return


def calc_depth_ref(R_key_to_ref, t_key_to_ref, x_key, depth_key):
    p = depth_key * to_homogeneous(x_key)
    q = transform(R_key_to_ref, t_key_to_ref, p)
    return q[2]


def coordinates_along_ref_epipolar(R_key_to_ref, t_key_to_ref,
                                   x_key, inv_depths):
    P = np.outer(1 / inv_depths, to_homogeneous(x_key))
    Q = transform(R_key_to_ref, t_key_to_ref, P)
    return pi(Q)


def coordinates_along_key_epipolar(x_key, t_key_to_ref, disparity):
    direction = normalize_length(x_key - pi(t_key_to_ref))
    return x_key + np.outer([-2, -1, 0, 1, 2], disparity * direction)


class InsufficientCoordinatesError(Exception):
    pass


def error_if_insufficient_coordinates(mask, min_coordinates):
    n_in_image = np.sum(mask)
    if n_in_image < min_coordinates:
        raise InsufficientCoordinatesError(
            "Insufficient number of coordinates "
            "sampled from the epipolar line. "
            "Required {}, but found {}.".format(min_coordinates, n_in_image)
        )


class DepthEstimator(object):
    def __init__(self, camera_model_key, camera_model_ref,
                 image_key, image_ref, pose_key_to_ref):
        self.camera_model_key = camera_model_key
        self.camera_model_ref = camera_model_ref
        self.image_key = image_key
        self.image_ref = image_ref
        self.pose_key_to_ref = pose_key_to_ref

    def __call__(self, u_key, min_depth, max_depth, prior_depth_key):
        R = self.pose_key_to_ref.rotation.as_matrix()
        t = self.pose_key_to_ref.t

        x_key = self.camera_model_key.normalize(u_key)

        disparity_ref = 0.01

        depth_ref = calc_depth_ref(R, t, x_key, prior_depth_key)
        inv_depth_ref = 1 / depth_ref
        inv_depth_key = 1 / prior_depth_key
        disparity_key = disparity_ref * (inv_depth_key / inv_depth_ref)

        # TODO check gradient along epipolar line
        epipolar_direction_key = normalize_length(x_key - pi(t))
        xs_key = coordinates_along_line(
            x_key, epipolar_direction_key,
            disparity_key * np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        )
        us_key = self.camera_model_key.unnormalize(xs_key)
        mask = is_in_image_range(us_key,
                                 [self.image_key.shape[0]-1,
                                  self.image_key.shape[1]-1])
        error_if_insufficient_coordinates(
            is_in_image_range(us_key, self.image_key.shape), len(xs_key)
        )

        # searching along the epipolar line with equal disparity means
        # searching the depth with equal inverse depth step
        # because inverse depth is approximately proportional to
        # the disparity

        # search from max to min
        x_ref_max = pi(transform(R, t, to_homogeneous(x_key) * max_depth))
        x_ref_min = pi(transform(R, t, to_homogeneous(x_key) * min_depth))
        epipolar_direction_ref = normalize_length(x_ref_min - x_ref_max)

        N = np.linalg.norm(x_ref_min - x_ref_max) / disparity_ref
        xs_ref = coordinates_along_line(x_ref_max, epipolar_direction_ref,
                                        disparity_ref * np.arange(N))
        us_ref = self.camera_model_ref.unnormalize(xs_ref)

        mask = is_in_image_range(us_ref,
                                 [self.image_ref.shape[0]-1,
                                  self.image_ref.shape[1]-1])
        error_if_insufficient_coordinates(mask, len(xs_key))
        xs_ref, us_ref = xs_ref[mask], us_ref[mask]

        intensities_ref = interpolate(self.image_ref, us_ref)
        intensities_key = interpolate(self.image_key, us_key)

        argmin = search_intensities(intensities_ref, intensities_key,
                                    calc_error)

        f = DepthFromTriangulation(Pose.identity(), self.pose_key_to_ref)
        key_depth, ref_depth = f(xs_key[2], xs_ref[argmin])
        return key_depth
