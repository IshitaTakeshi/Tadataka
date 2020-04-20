from tadataka.coordinates import image_coordinates
from tadataka.utils import is_in_image_range


def coordinates_(warp10, depth_map0):
    us0 = image_coordinates(depth_map0.shape)
    depths0 = depth_map0.flatten()
    us1, depths1 = warp10(us0, depths0)
    return us0, us1, depths0, depths1


def warp_coordinates(warp10, depth_map0):
    us0, us1, depths0, depths1 = coordinates_(warp10, depth_map0)
    mask = is_in_image_range(us1, depth_map0.shape)
    return us0[mask], us1[mask], depths0[mask], depths1[mask]
