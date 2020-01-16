import numpy as np
from tadataka.utils import is_in_image_range
from tadataka.matrix import to_homogeneous
from tadataka.projection import pi
from tadataka.rigid_transform import transform


def normalize_length(v):
    return v / np.linalg.norm(v)


def inverse_projection(camera_model, x, depth):
    return depth * to_homogeneous(x)


def calc_epipolar_direction(coordinate, t, K):
    return normalize_length(coordinate - projection(t, K))


# TODO use numba for acceleration
def convolve(a, b, calc_error):
    # reverese b because the second argument is reversed in the computation
    N = len(b)
    return np.array([calc_error(a[i:i+N], b) for i in range(len(a)-N+1)])


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


def search_intensities(ref_intensities, key_intensities, calc_error):
    errors = convolve(ref_intensities, key_intensities, calc_error)
    return np.argmin(errors)


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


class DepthEstimator(object):
    def __init__(self, camera_model_key, camera_model_ref,
                 image_key, image_ref, pose_key_to_ref):
        self.camera_model_key = camera_model_key
        self.camera_model_ref = camera_model_ref
        self.image_key = image_key
        self.image_ref = image_ref
        self.R = pose_key_to_ref.rotation.as_matrix()
        self.t = pose_key_to_ref.t

    def __call__(self, u_key, min_depth, max_depth, prior_depth_key):
        x_key = self.camera_model_key.normalize(u_key)
        print(x_key)

        disparity_ref = 0.01

        depth_ref = calc_depth_ref(self.R, self.t, x_key, prior_depth_key)
        inv_depth_ref = 1 / depth_ref
        inv_depth_key = 1 / prior_depth_key
        disparity_key = disparity_ref * (inv_depth_key / inv_depth_ref)

        # TODO check gradient along epipolar line
        epipolar_direction_key = normalize_length(x_key - pi(self.t))
        xs_key = coordinates_along_line(
            x_key, epipolar_direction_key,
            disparity_key * np.array([-2, -1, 0, 1, 2])
        )

        us_key = self.camera_model_key.unnormalize(xs_key)
        # if not np.all(is_in_image_range(us_key, self.image_key.shape)):
        #     raise ValueError(
        #         "All coordinates sampled from keyframe epipolar line "
        #         "have to be within the keyframe image range"
        #     )

        # searching along epipolar line with equal disparity means
        # searching the depth with equal inverse depth step
        # because inverse depth is approximately proportional to
        # the disparity

        x_ref_max = pi(transform(self.R, self.t, to_homogeneous(x_key) * max_depth))
        x_ref_min = pi(transform(self.R, self.t, to_homogeneous(x_key) * min_depth))
        x_ref = pi(transform(self.R, self.t, to_homogeneous(x_key) * prior_depth_key))
        epipolar_direction_ref = normalize_length(x_ref_max - x_ref_min)
        N = np.linalg.norm(x_ref_max - x_ref_min) / disparity_ref
        xs_ref = coordinates_along_line(
            x_ref_min, epipolar_direction_ref,
            disparity_ref * np.arange(N)
        )
        us_ref = self.camera_model_ref.unnormalize(xs_ref)
        print(us_ref)
        from matplotlib import pyplot as plt
        plt.subplot(121)
        plt.imshow(self.image_key)
        plt.scatter(u_key[0], u_key[1])
        plt.subplot(122)
        plt.imshow(self.image_ref)
        plt.scatter(us_ref[:, 0], us_ref[:, 1])
        plt.show()

        mask = is_in_image_range(us_ref, self.image_ref.shape)
        if np.sum(mask) < min_ref_coordinates:
            raise ValueError("Insufficient number of coordinates "
                             "sampled from the reference epipolar line")
        xs_ref, us_ref = xs_ref[mask], us_ref[mask]

        intensities_ref = interpolate(self.image_ref, us_ref)
        intensities_key = interpolate(self.image_key, us_key)

        argmin = search_intensities(ref_intensities, key_intensities, calc_error)
        [key_depth, ref_depth] = estimate_depths(self.R, self.t,
                                                 key_coordinates[2],
                                                 ref_coordinates[argmin])
        return key_depth
