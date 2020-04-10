from skimage.color import rgb2gray
from skimage.transform import resize
from tqdm import tqdm

from tadataka.metric import PhotometricError
from tadataka.pose import WorldPose
from tadataka.dataset import TumRgbdDataset, NewTsukubaDataset
from tadataka.vo.dvo import PoseChangeEstimator
from tadataka.camera import CameraModel
from tadataka import camera
from tests.dataset.path import new_tsukuba
from examples.plot import plot_trajectory, plot_warp
import numpy as np
np.set_printoptions(suppress=True)


def to_perspective(camera_model):
    return CameraModel(camera_model.camera_parameters,
                       distortion_model=None)


def get(frame):
    camera_model = to_perspective(frame.camera_model)
    I = rgb2gray(frame.image)
    D = frame.depth_map
    return camera_model, I, D


def dvo(camera_model0, camera_model1, I0, D0, I1, weights):
    estimator = PoseChangeEstimator(camera_model0, camera_model1,
                                    n_coarse_to_fine=6)
    return estimator(I0, D0, I1, weights)


def main():
    dataset = NewTsukubaDataset("datasets/NewTsukubaStereoDataset")
    frames = [fl for fl, fr in dataset[0:500]]
    # dataset = TumRgbdDataset("datasets/rgbd_dataset_freiburg1_desk",
    #                          which_freiburg=1)
    # frames = dataset[::3]

    trajectory_true = []
    trajectory_pred = []
    pose_pred = WorldPose.identity()
    pose_true = WorldPose.identity()
    for i in tqdm(range(len(frames)-1)):
        frame0_, frame1_ = frames[i], frames[i+1]

        camera_model0, I0, D0 = get(frame0_)
        camera_model1, I1, __ = get(frame1_)
        error = PhotometricError(camera_model0, camera_model1,
                                 I0, D0, I1)

        pose10_true = frame1_.pose.inv() * frame0_.pose
        pose10_pred = dvo(camera_model0, camera_model1, I0, D0, I1, None)

        print("pose10_true :", pose10_true)
        print("pose10_pred :", pose10_pred)

        print("error identity = {:.6f}".format(error(WorldPose.identity())))
        print("error true     = {:.6f}".format(error(pose10_true)))
        print("error pred     = {:.6f}".format(error(pose10_pred)))

        if False:  # error_pred > error_true:
            from examples.plot import plot_warp
            plot_warp(LocalWarp2D(camera_model0, camera_model1, pose10_true),
                      I0, D0, I1)
            plot_warp(LocalWarp2D(camera_model0, camera_model1, pose10_pred),
                      I0, D0, I1)
        pose_pred = pose_pred * pose10_pred.inv()
        pose_true = pose_true * pose10_true.inv()
        trajectory_pred.append(pose_pred.t)
        trajectory_true.append(pose_true.t)

    plot_trajectory(np.array(trajectory_true), np.array(trajectory_pred))

main()
