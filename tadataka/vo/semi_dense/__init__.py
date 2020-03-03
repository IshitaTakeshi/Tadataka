import numpy as np

from skimage.color import rgb2gray
from tadataka.projection import Warp
from tadataka.pose import calc_relative_pose
from tadataka.vo.semi_dense.semi_dense import InverseDepthEstimator, InverseDepthSearchRange
from tadataka.vo.semi_dense.age import increment_age
from tadataka.coordinates import image_coordinates
from tadataka.vo.semi_dense.fusion import fusion


class SemiDenseVO(object):
    def __init__(self):
        self.inv_depth_map = None
        self.variance_map = None
        self.images = []
        self.poses = []
        self.camera_models = []
        self.age = None

    def estimate(self, keyframe):
        image = rgb2gray(keyframe.image)
        pose = keyframe.pose
        camera_model = keyframe.camera_model
        # depth_map have to be replaced with the estimated one
        depth_map = keyframe.depth_map

        if len(self.images) == 0:
            self.images.append(image)
            self.poses.append(pose)
            self.camera_models.append(camera_model)

            image_shape = image.shape[0:2]
            self.age = np.zeros(image_shape, dtype=np.int64)
            self.inv_depth_map = np.ones(image_shape)
            self.variance_map = np.ones(image_shape)
            return np.nan

        warp = Warp(self.camera_models[-1], camera_model,
                    calc_relative_pose(self.poses[-1], pose))
        self.age = increment_age(self.age, warp, depth_map)
        estimator = InverseDepthMapEstimator(
            image, pose, camera_model,
        )

        estimator(self.poses, self.camera_models, self.images, self.age,
                  self.inv_depth_map, self.variance_map)

        self.images.append(image)
        self.poses.append(pose)
        self.camera_models.append(camera_model)
        # self.inv_depth_map, self.variance_map = fusion(
        #     inv_depth_map, self.inv_depth_map,
        #     variance_map, self.variance_map
        # )


class InverseDepthMapEstimator(object):
    def __init__(self, image_key, pose_key, camera_model_key):
        self.image_shape = image_key.shape[0:2]
        self.pose_key = pose_key
        self.estimator = InverseDepthEstimator(
            image_key, camera_model_key,
            sigma_i=0.01, sigma_l=0.02,
            step_size_ref=0.01,
            min_gradient=1.0
        )
        self.search_range = InverseDepthSearchRange(
            min_inv_depth=0.05, max_inv_depth=10.0
        )

    def __call__(self, poses_ref, camera_models_ref, images_ref, pixel_age,
                 prior_inv_depth_map, prior_variance_map):
        from tqdm import tqdm
        from matplotlib import pyplot as plt

        # plt.title("Pixel age")
        # plt.imshow(100 * pixel_age, cmap="gray")
        # plt.show()

        inv_depth_map = np.empty(self.image_shape)
        variance_map = np.empty(self.image_shape)
        for u_key in tqdm(image_coordinates(self.image_shape)):
            x, y = u_key
            age = pixel_age[y, x]
            if age == 0:
                # no reference frame
                continue

            pose_ref = poses_ref[-age]
            camera_model_ref = camera_models_ref[-age]
            image_ref = images_ref[-age]

            pose_key_to_ref = calc_relative_pose(self.pose_key, pose_ref)

            prior_inv_depth = prior_inv_depth_map[y, x]
            prior_variance = prior_variance_map[y, x]

            inv_depth_map[y, x], variance_map[y, x] = self.estimator(
                pose_key_to_ref, camera_model_ref, image_ref,
                u_key, prior_inv_depth,
                self.search_range(prior_inv_depth, prior_variance)
            )


        plt.subplot(236)
        plt.title("Pixel age")
        plt.imshow(pixel_age, cmap="gray")
        plt.show()

        plt.subplot(221)
        plt.title("Reference frame")
        plt.imshow(images_ref[-1], cmap="gray")

        plt.subplot(222)
        plt.title("Keyframe")
        plt.imshow(self.estimator.image_key, cmap="gray")

        plt.subplot(223)
        plt.title("Keyframe inverse depth")
        plt.imshow(inv_depth_map, cmap="gray")

        plt.subplot(224)
        plt.title("Uncertaintity of inverse depth")
        plt.imshow(variance_map, cmap="gray")

        plt.show()

        return inv_depth_map, variance_map
