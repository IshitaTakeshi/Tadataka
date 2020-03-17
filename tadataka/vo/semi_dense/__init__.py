import numpy as np

from skimage.color import rgb2gray
# from tadataka.projection import Warp
from tadataka.pose import calc_relative_pose
from tadataka.vo.semi_dense.common import invert_depth
from tadataka.vo.semi_dense.semi_dense import InverseDepthEstimator, InverseDepthSearchRange
from tadataka.vo.semi_dense.age import increment_age
from tadataka.coordinates import image_coordinates
from tadataka.vo.semi_dense.fusion import fusion
from tadataka.vo.semi_dense.frame import Frame


def plot(image, pixel_age, inv_depth_map, variance_map):
    from matplotlib import pyplot as plt
    import matplotlib

    def image_with_bar(fig, ax, image, n_ticks=5):
        cax = ax.imshow(image, cmap=matplotlib.cm.gray)
        ticks = np.linspace(np.nanmin(image), np.nanmax(image), n_ticks)
        cbar = fig.colorbar(cax, ticks=ticks)
        cbar.ax.set_yticklabels(["{:.3f}".format(e) for e in ticks])

    fig = plt.figure()

    ax = fig.add_subplot(131)
    ax.set_title("Keyframe")
    ax.imshow(image, cmap="gray")

    ax = fig.add_subplot(232)
    ax.set_title("Pixel age")
    image_with_bar(fig, ax, pixel_age, n_ticks=3)

    ax = fig.add_subplot(233)
    ax.set_title("Keyframe inverse depth")
    image_with_bar(fig, ax, inv_depth_map)

    ax = fig.add_subplot(235)
    ax.set_title("Depth map")
    image_with_bar(fig, ax, invert_depth(inv_depth_map))

    ax = fig.add_subplot(236)
    ax.set_title("Uncertaintity of inverse depth")
    image_with_bar(fig, ax, variance_map)

    plt.show()


class DepthMapUpdate(object):
    def __init__(self, keyframe, refframes, pixel_age):
        self.estimator = InverseDepthMapEstimator(
            keyframe, refframes, pixel_age
        )

    def __call__(self, prior_inv_depth_map, prior_variance_map):
        inv_depth_map, variance_map = self.estimator(prior_inv_depth_map,
                                                     prior_variance_map)
        return fusion(inv_depth_map, prior_inv_depth_map,
                      variance_map, prior_variance_map)


class DepthMapPropagation(object):
    def __init__(self, pose01):
        self.tz = pose01.t[2]

    def __call__(self, inv_depth_map0, variance_map0):
        inv_depth_map1 = new_inverse_depth_map(inv_depth_map0, self.tz)
        variance_map1 = new_variance_map(inv_depth_map0, inv_depth_map1,
                                         variance_map0)
        return inv_depth_map1, variance_map1


class SemiDenseVO(object):
    def __init__(self):
        self.inv_depth_map = None
        self.variance_map = None
        self.refframes = []
        self.age = None

    def estimate(self, keyframe_):
        keyframe = Frame(
            keyframe_.camera_model,
            rgb2gray(keyframe_.image),
            keyframe_.pose.to_local()
        )

        if len(self.refframes) == 0:
            self.refframes.append(keyframe)

            image_shape = keyframe.image.shape
            self.age = np.zeros(image_shape, dtype=np.int64)
            self.inv_depth_map = np.ones(image_shape)
            self.variance_map = np.ones(image_shape)

            plot(keyframe.image, self.age,
                 self.inv_depth_map, self.variance_map)

            return

        inv_depth_map, variance_map = self.inv_depth_map, self.variance_map

        last = self.refframes[-1]
        pose01 = calc_relative_pose(last.pose, keyframe.pose)

        warp = Warp(last.camera_model, keyframe.camera_model, pose01)
        self.age = increment_age(self.age, warp, invert_depth(inv_depth_map))

        propagate = DepthMapPropagation(pose01)
        inv_depth_map, variance_map = propagate(inv_depth_map, variance_map)

        update = DepthMapUpdate(keyframe, self.refframes, self.age)
        inv_depth_map, variance_map = update(inv_depth_map, variance_map)

        self.inv_depth_map, self.variance_map = inv_depth_map, variance_map
        self.refframes.append(keyframe)

        plot(keyframe.image, self.age,
             self.inv_depth_map, self.variance_map)


class InverseDepthMapEstimator(object):
    def __init__(self, keyframe, refframes, pixel_age):
        self.image_shape = keyframe.image.shape[0:2]
        self.pose_key = keyframe.pose
        self.refframes = refframes
        self.pixel_age = pixel_age
        self.estimator = InverseDepthEstimator(
            keyframe.image, keyframe.camera_model,
            sigma_i=0.01, sigma_l=0.02,
            step_size_ref=0.01,
            min_gradient=1.0
        )

    def __call__(self, prior_inv_depth_map, prior_variance_map):
        from tqdm import tqdm

        # plt.title("Pixel age")
        # plt.imshow(100 * pixel_age, cmap="gray")
        # plt.show()

        inv_depth_map = np.empty(self.image_shape)
        variance_map = np.empty(self.image_shape)
        for u_key in tqdm(image_coordinates(self.image_shape)):
            x, y = u_key
            age = self.pixel_age[y, x]
            if age == 0:
                # no reference frame
                continue

            refframe = self.refframes[-age]

            pose_key_to_ref = calc_relative_pose(self.pose_key, refframe.pose)

            inv_depth_map[y, x], variance_map[y, x] = self.estimator(
                pose_key_to_ref, refframe.camera_model, refframe.image,
                u_key, prior_inv_depth_map[y, x], prior_variance_map[y, x]
            )

        return inv_depth_map, variance_map
