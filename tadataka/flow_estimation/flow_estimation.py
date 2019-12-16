from autograd import numpy as np

from skimage.transform import AffineTransform
from tadataka import irls


def estimate_affine_transform(keypoints0, keypoints1):
    assert(keypoints0.shape == keypoints1.shape)

    keypoints0 = np.column_stack((keypoints0, np.ones(keypoints0.shape[0])))
    params0 = irls.fit(keypoints0, keypoints1[:, 0])
    params1 = irls.fit(keypoints0, keypoints1[:, 1])

    M = np.identity(3)
    M[0] = params0
    M[1] = params1
    return AffineTransform(M)
