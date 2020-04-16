import numpy as np
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize

from tqdm import tqdm
from tadataka.camera import CameraModel
from tadataka.rigid_motion import LeastSquaresRigidMotion
from tadataka.vo.semi_dense.frame_selection import ReferenceSelector
from tadataka.coordinates import image_coordinates
from tadataka.dataset import TumRgbdDataset
from tadataka.vo.semi_dense.frame import Frame
from tadataka.vo.semi_dense.age import increment_age
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.feature import extract_features, Matcher
from tadataka.vo.semi_dense.flag import ResultFlag as FLAG
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.vo.semi_dense.propagation import propagate
from tadataka.pose import WorldPose, estimate_pose_change
from tadataka.vo.dvo import PoseChangeEstimator
from tadataka.matrix import to_homogeneous
from tadataka.warp import Warp2D
from tadataka.rigid_transform import transform, Transform
from tadataka.dataset import NewTsukubaDataset
from tests.dataset.path import new_tsukuba

from examples.plot import plot_depth, plot_prior
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tadataka.vo.semi_dense.semi_dense import InverseDepthMapEstimator


def update(keyframe, ref_selector, prior_inv_depth_map, prior_variance_map):
    estimator = InverseDepthMapEstimator(keyframe, sigma_i=0.01, sigma_l=0.02,
                                         step_size_ref=0.01, min_gradient=20.0)

    inv_depth_map, variance_map, flag_map = estimator(
        ref_selector, prior_inv_depth_map, prior_variance_map
    )

    inv_depth_map, variance_map = fusion(prior_inv_depth_map, inv_depth_map,
                                         prior_variance_map, variance_map)

    return inv_depth_map, variance_map, flag_map


def resize_camera_model(cm, scale):
    from tadataka.camera import CameraParameters
    return CameraModel(
        CameraParameters(cm.camera_parameters.focal_length * scale,
                         cm.camera_parameters.offset * scale),
        cm.distortion_model
    )


def dvo_(cm0, cm1, I0, D0, I1, W, scale=1.0):
    estimator = PoseChangeEstimator(resize_camera_model(cm0, scale),
                                    resize_camera_model(cm1, scale),
                                    n_coarse_to_fine=3)

    shape = (int(I0.shape[0] * scale), int(I0.shape[1] * scale))
    return estimator(resize(I0, shape), resize(D0, shape),
                     resize(I1, shape), resize(W, shape))


def dvo(camera_model0, camera_model1, image0, image1,
        inv_depth_map, variance_map):
    return dvo_(camera_model0, camera_model1,
                image0, invert_depth(inv_depth_map),
                image1, invert_depth(variance_map))


def init_pose10(camera_model0, camera_model1, image0, image1):
    features0 = extract_features(image0)
    features1 = extract_features(image1)
    matches01 = Matcher()(features0, features1)
    keypoints0 = features0.keypoints[matches01[:, 0]]
    keypoints1 = features1.keypoints[matches01[:, 1]]
    from tadataka.plot import plot_matches
    plot_matches(image0, image1,
                 features0.keypoints, features1.keypoints, matches01)
    local_pose10 = estimate_pose_change(camera_model0.normalize(keypoints0),
                                        camera_model1.normalize(keypoints1))

    from tadataka.triangulation import TwoViewTriangulation
    triangulation = TwoViewTriangulation(WorldPose.identity(), local_pose10)
    points, depths = triangulation.triangulate(keypoints0, keypoints1)
    from tadataka.plot import plot_map
    plot_map([WorldPose.identity(), local_pose10.to_world()], points)

    pose10 = local_pose10.to_world()
    return pose10


def to_perspective(camera_model):
    return CameraModel(
        camera_model.camera_parameters,
        distortion_model=None
    )


def main():
    dataset = TumRgbdDataset("datasets/rgbd_dataset_freiburg1_desk",
                             which_freiburg=1)
    camera_model0, pose_true0, image0, depth_map_true0 = dataset[200+0*10]
    camera_model1, pose_true1, image1, depth_map_true1 = dataset[200+1*10]
    camera_model2, pose_true2, image2, depth_map_true2 = dataset[200+2*10]
    camera_model3, pose_true3, image3, depth_map_true3 = dataset[200+3*10]
    camera_model0, image0 = to_perspective(camera_model0), rgb2gray(image0)
    camera_model1, image1 = to_perspective(camera_model1), rgb2gray(image1)
    camera_model2, image2 = to_perspective(camera_model2), rgb2gray(image2)
    camera_model3, image3 = to_perspective(camera_model3), rgb2gray(image3)

    # np.random.uniform(0.1, 10.0, image0.shape)
    inv_depth_map0 = invert_depth(depth_map_true0)
    variance_map0 = 10.0 * np.ones(image0.shape)
    age_map0 = np.zeros(image0.shape, dtype=np.int64)

    # pose0 = WorldPose.identity()
    # pose10 = init_pose10(camera_model0, camera_model1, image0, image1)
    # pose1 = pose10 * pose0
    pose0, pose1 = pose_true0, pose_true1
    frame0 = Frame(camera_model0, image0, pose0)

    # plot_depth(image0, age_map0, np.zeros(inv_depth_map0.shape, dtype=np.int64),
    #            depth_map_true0, invert_depth(inv_depth_map0), variance_map0)

    # from tadataka.plot import plot_map
    # plot_map([pose0, pose1], np.zeros((0, 3)))

    warp10 = Warp2D(camera_model0, camera_model1, pose0, pose1)
    age_map1 = increment_age(age_map0, warp10, inv_depth_map0)
    frame1 = Frame(camera_model1, image1, pose1)

    prior_inv_depth_map1, prior_variance_map1 = propagate(
        warp10, inv_depth_map0, variance_map0
    )
    plot_depth(image1, age_map1, np.zeros(image1.shape, dtype=np.int64),
               depth_map_true1,
               invert_depth(prior_inv_depth_map1), prior_variance_map1)

    inv_depth_map1, variance_map1, flag_map1 = update(
        frame1, ReferenceSelector([frame0], age_map1),
        prior_inv_depth_map1, prior_variance_map1
    )
    plot_depth(image1, age_map1, flag_map1,
               depth_map_true1, invert_depth(inv_depth_map1), variance_map1)

    pose21 = dvo(camera_model1, camera_model2, image1, image2,
                 inv_depth_map1, variance_map1)

    pose2 = pose21 * pose1

    warp21 = Warp2D(camera_model1, camera_model2, WorldPose.identity(), pose21)
    age_map2 = increment_age(age_map1, warp21, inv_depth_map1)
    frame2 = Frame(camera_model2, image2, pose2)

    prior_inv_depth_map2, prior_variance_map2 = propagate(
        warp21, inv_depth_map1, variance_map1
    )

    inv_depth_map2, variance_map2, flag_map2 = update(
        frame2, ReferenceSelector([frame0, frame1], age_map2),
        prior_inv_depth_map2, prior_variance_map2
    )
    plot_depth(image2, age_map2, flag_map2,
               depth_map_true2, invert_depth(inv_depth_map2), variance_map2)

    pose32 = dvo(camera_model2, camera_model3, image2, image3,
                 inv_depth_map2, variance_map2)

    pose3 = pose32 * pose2

    warp32 = Warp2D(camera_model2, camera_model3, WorldPose.identity(), pose32)
    age_map3 = increment_age(age_map2, warp32, inv_depth_map2)
    frame3 = Frame(camera_model3, image3, pose3)

    prior_inv_depth_map3, prior_variance_map3 = propagate(
        warp32, inv_depth_map2, variance_map2
    )
    inv_depth_map3, variance_map3, flag_map3 = update(
        frame3, ReferenceSelector([frame0, frame1, frame2], age_map3),
        prior_inv_depth_map3, prior_variance_map3
    )
    plot_depth(image3, age_map3, flag_map3,
               depth_map_true3, invert_depth(inv_depth_map3), variance_map3)



def main_():
    dataset = TumRgbdDataset("datasets/rgbd_dataset_freiburg1_desk",
                             which_freiburg=1)

    frames = dataset[0:200:10]

    inv_depth_map = invert_depth(frames[0].depth_map)
    variance_map = 10.0 * np.ones(frames[0].depth_map.shape)

    trajectory_true = []
    trajectory_pred = []
    for i in range(len(frames)):
        camera_model, pose_true, image, depth_map_true = frames[i]
        Frame(camera_model, image, pose_pred)

        pose10_true = frame1_.pose.inv() * frame0_.pose
        pose_true = pose10_true * pose_true

        trajectory_pred.append(pose_pred.t)
        trajectory_true.append(pose_true.t)

    trajectory_true = np.array(trajectory_true)
    trajectory_pred = np.array(trajectory_pred)
    R, t, s = LeastSquaresRigidMotion(trajectory_pred, trajectory_true).solve()
    trajectory_pred = Transform(R, t, s)(trajectory_pred)
    print("MSE: ", np.power(trajectory_pred - trajectory_true, 2).mean())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trajectory_pred[:, 0], trajectory_pred[:, 1], trajectory_pred[:, 2],
            label="pred")
    ax.plot(trajectory_true[:, 0], trajectory_true[:, 1], trajectory_true[:, 2],
            label="true")
    plt.legend()
    plt.show()

main()
