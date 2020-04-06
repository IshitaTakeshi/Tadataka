import numpy as np
from tadataka.interpolation import interpolation2d_
from tadataka.pose import LocalPose
from tadataka.projection import inv_pi, pi
from tadataka.coordinates import image_coordinates
from tadataka.utils import is_in_image_range
from tadataka.rigid_transform import transform

def calc_error_(v1, v2):
    return np.mean(np.power(v1 - v2, 2))


def photometric_error(camera_model0, camera_model1, pose10: LocalPose,
                      gray_image0, depth_map0, gray_image1):
    us0 = image_coordinates(depth_map0.shape)
    xs0 = camera_model0.normalize(us0)
    P0 = inv_pi(xs0, depth_map0.flatten())
    P1 = transform(pose10.R, pose10.t, P0)
    xs1 = pi(P1)
    us1 = camera_model1.unnormalize(xs1)

    mask = is_in_image_range(us1, depth_map0.shape)

    i0 = gray_image0[us0[mask, 1], us0[mask, 0]]
    i1 = interpolation2d_(gray_image1, us1[mask])

    return calc_error_(i0, i1)
