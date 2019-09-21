from autograd import numpy as np


def solve_pnp(points, keypoints):
    # TODO make independent from cv2
    import cv2

    retval, rvec, tvec = cv2.solvePnP(points.astype(np.float64),
                                      keypoints.astype(np.float64),
                                      np.identity(3), np.zeros(4))
    rvec = rvec.flatten()
    tvec = tvec.flatten()
    return rvec, tvec
