import numpy as np
from scipy.spatial.transform import Rotation

from tadataka.pose import Pose
from tadataka.vo.semi_dense.reference import relative_


def test_relative():
    pose_wr = Pose(Rotation.from_rotvec(np.random.random(3)),
                   np.random.random(3))
    pose_wk = Pose(Rotation.from_rotvec(np.random.random(3)),
                   np.random.random(3))
    pose_rk = relative_(pose_wr, pose_wk)
    assert(pose_wr * pose_rk == pose_wk)
