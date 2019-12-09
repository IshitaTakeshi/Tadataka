from autograd import numpy as np

from tadataka.transform import AffineTransform
from tadataka import irls


def estimate_affine_transform(keypoints0, keypoints1):
    keypoints0 = np.column_stack((keypoints0, np.ones(keypoints0.shape[0])))
    params0 = irls.fit(keypoints0, keypoints1[:, 0])
    params1 = irls.fit(keypoints0, keypoints1[:, 1])

    A0, b0 = params0[0:2], params0[2]
    A1, b1 = params1[0:2], params1[2]

    A = np.vstack((A0, A1))
    b = np.array([b0, b1])
    return AffineTransform(A, b)
