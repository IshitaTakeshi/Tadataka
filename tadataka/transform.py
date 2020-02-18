from tadataka.interpolation import interpolation
from tadataka.vo.dvo.projection import reprojection


def warp(camera_parameters, I1, D0, G):
    # this function samples pixels in I1 and project them to
    # I0 coordinate system

    # 'reprojection' transforms I0 coordinates to
    # the corresponding coordinates in I1

    # 'Q' has pixel coordinates in I1 coordinate system, but each pixel
    # coordinate is corresponding to the one in I0
    # Therefore image pixels sampled by 'Q' represents I1 transformed into
    # I0 coordinate system

    # 'G' describes the transformation from I0 coordinate system to
    # I1 coordinate system

    Q, mask = reprojection(camera_parameters, D0, G)

    warped_image = interpolation(I1, Q)
    warped_image = warped_image.reshape(D0.shape)

    return warped_image, mask
