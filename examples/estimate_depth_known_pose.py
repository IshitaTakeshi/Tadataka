import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from tqdm import tqdm

from tadataka import camera
from tadataka.pose import Pose
from tadataka.camera import CameraModel, CameraParameters
from tadataka.gradient import grad_x, grad_y
from tadataka.warp import warp2d, Warp2D
from tadataka.dataset import NewTsukubaDataset, TumRgbdDataset
from tadataka.vo.semi_dense.age import AgeMap, increment_age
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.vo.semi_dense.frame import ReferenceSelector
from tadataka.numeric import safe_invert
from tadataka.metric import PhotometricError
from tadataka.warp import LocalWarp2D
from tadataka.vo.semi_dense.propagation import Propagation
from tadataka.vo.semi_dense.gradient import GradientImage

from tadataka.vo.semi_dense.semi_dense import (
    InvDepthEstimator, InvDepthMapEstimator, AgeDependentValues
)
from tadataka.vo.semi_dense.regularization import regularize
from examples.plot import plot_depth, plot_trajectory


def plot_reprojection(keyframe, refframe):
    px, py = [200, 400]
    p = keyframe.camera_model.normalize(np.array([px, py], dtype=np.float64))
    q = warp2d((keyframe.pose.R, keyframe.pose.t),
               (refframe.pose.R, refframe.pose.t),
               p, keyframe.depth_map[py, px])
    qx, qy = refframe.camera_model.unnormalize(q)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(keyframe.image)
    ax.scatter(px, py)

    ax = fig.add_subplot(122)
    ax.imshow(refframe.image)
    ax.scatter(qx, qy)

    plt.show()


def to_perspective(camera_model):
    return CameraModel(camera_model.camera_parameters,
                       distortion_model=None)


def plot_refframes(refframes, keyframe, age_map):
    N = len(refframes) + 2

    fig = plt.figure()
    for i, frame in enumerate(refframes):
        ax = fig.add_subplot(1, N, i+1)
        ax.set_title(f"refframe {i}")
        ax.imshow(frame.image, cmap="gray")

    ax = fig.add_subplot(1, N, len(refframes)+1)
    ax.set_title("keyframe")
    ax.imshow(keyframe.image, cmap="gray")

    ax = fig.add_subplot(1, N, len(refframes)+2)
    ax.set_title("pixel age")
    ax.imshow(age_map, cmap="gray")

    plt.show()


def get(frame, scale=1.0):
    camera_model = to_perspective(frame.camera_model)
    camera_model = camera.resize(camera_model, scale)

    image = rgb2gray(frame.image)
    shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))
    image = resize(image, shape)
    return camera_model, image, frame.pose


def dvo(camera_model0, camera_model1, image0, image1, depth_map0, weights):
    estimator = PoseChangeEstimator(camera_model0, camera_model1,
                                    n_coarse_to_fine=7)
    return estimator(image0, depth_map0, image1, weights)


def update(estimator, prior_depth_map, prior_variance_map,
           reference_selector):
    prior_inv_depth_map = safe_invert(prior_depth_map)
    inv_depth_map, variance_map, flag_map = estimator(
        prior_inv_depth_map, prior_variance_map,
        reference_selector
    )

    inv_depth_map, variance_map = fusion(prior_inv_depth_map, inv_depth_map,
                                         prior_variance_map, variance_map)

    inv_depth_map = regularize(inv_depth_map, variance_map)

    return safe_invert(inv_depth_map), variance_map, flag_map


from tests.dataset.path import new_tsukuba


def to_relative(ref, pose_wk):
    camera_model_ref, image_ref, pose_wr = ref
    pose_rk = pose_wr.inv() * pose_wk
    return (camera_model_ref, image_ref, pose_rk.T)


estimator_params = {
    "sigma_i": 0.1,
    "sigma_l": 0.2,
    "step_size_ref": 0.005,
    "min_gradient": 0.2
}


class Estimator(object):
    def __init__(self, first_frame):
        _, image, _ = first_frame

        self.refframes = [first_frame]

        self.depth_range = [60, 1000]
        default_variance = 100
        self.propagate = Propagation(default_depth=200,
                                     default_variance=default_variance,
                                     uncertaintity_bias=1.0)
        self.depth_map = np.random.uniform(*self.depth_range, image.shape)
        self.variance_map = default_variance * np.ones(image.shape)
        self.age_map = np.zeros(image.shape, dtype=np.int64)

    def make_estimator(self, camera_model_key, image_key):
        min_depth, max_depth = self.depth_range
        inv_depth_range = [1 / max_depth, 1 / min_depth]
        return InvDepthMapEstimator(
            InvDepthEstimator(camera_model_key, image_key,
                              inv_depth_range, **estimator_params)
        )

    def __call__(self, current_keyframe):
        last_camera_model, _, last_pose = self.refframes[-1]
        last_depth_map = self.depth_map
        last_variance_map = self.variance_map

        current_camera_model, current_image, current_pose = current_keyframe
        warp_cl = Warp2D(last_camera_model, current_camera_model,
                         last_pose, current_pose)
        self.age_map = increment_age(self.age_map, warp_cl, last_depth_map)
        current_depth_map, current_variance_map = self.propagate(
            warp_cl, last_depth_map, last_variance_map
        )

        assert(self.age_map.max() == len(self.refframes))

        reference_selector = AgeDependentValues(
            self.age_map,
            [to_relative(f, current_pose) for f in self.refframes]
        )
        self.depth_map, self.variance_map, flag_map = update(
            self.make_estimator(current_camera_model, current_image),
            current_depth_map, current_variance_map, reference_selector
        )

        self.refframes.append(current_keyframe)
        return self.depth_map, self.variance_map, flag_map


def main():
    dataset = NewTsukubaDataset("datasets/NewTsukubaStereoDataset")
    frames = [dataset[i][0] for i in range(200, 250, 5)]

    estimator = Estimator(get(frames[0]))

    for frame in frames[1:]:
        depth_map, variance_map, flag_map = estimator(get(frame))
        plot_depth(frame.image, np.zeros(frame.depth_map.shape),
                   flag_map, frame.depth_map,
                   depth_map, variance_map)

    # plot_trajectory(np.array(trajectory_true), np.array(trajectory_pred))

main()
