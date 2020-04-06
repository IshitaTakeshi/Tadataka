from tadataka.vo.semi_dense.common import invert_depth
from tadataka.coordinates import image_coordinates
from tadataka.utils import is_in_image_range


def substitute(array2d, us, values):
    assert(us.shape[0] == values.shape[0])
    xs, ys = us[:, 0], us[:, 1]
    array2d[ys, xs] = values
    return array2d


def get(array2d, us):
    xs, ys = us[:, 0], us[:, 1]
    return array2d[ys, xs]


def coordinates_(warp10, depth_map0):
    us0 = image_coordinates(depth_map0.shape)
    depths0 = depth_map0.flatten()
    us1, depths1 = warp10(us0, depths0)
    return us0, us1, depths0, depths1


def coordinates(warp10, inv_depth_map0):
    depth_map0 = invert_depth(inv_depth_map0)
    us0, us1, depths0, depths1 = coordinates_(warp10, depth_map0)

    mask = is_in_image_range(us1, depth_map0.shape)
    return (us0[mask], us1[mask],
            invert_depth(depths0[mask]), invert_depth(depths1[mask]))
