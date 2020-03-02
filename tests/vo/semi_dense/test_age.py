import numpy as np
from matplotlib import pyplot as plt

from tadataka.coordinates import image_coordinates
from tadataka.utils import is_in_image_range
from tadataka.projection import Warp
from tadataka.pose import calc_relative_pose
from tadataka.dataset import NewTsukubaDataset, TumRgbdDataset


dataset = TumRgbdDataset("datasets/rgbd_dataset_freiburg1_xyz",
                         which_freiburg=1)

dataset = NewTsukubaDataset("datasets/NewTsukubaStereoDataset")


def test_pixel_age():
    start, end = 280, 289
    frames = [fl for fl, fr in dataset[start:end]]
    image_shape = frames[0].image.shape[0:2]
    age0 = np.zeros(image_shape, dtype=np.uint8)

    for i in range(len(frames)-1):
        frame0, frame1 = frames[i], frames[i+1]
        pose01 = calc_relative_pose(frame0.pose.to_local(),
                                    frame1.pose.to_local())

        warp = Warp(frame0.camera_model, frame1.camera_model, pose01)
        us0 = image_coordinates(frame0.image.shape)
        depths0 = frame0.depth_map.flatten()
        us1 = warp(us0, depths0)
        mask = is_in_image_range(us1, frame0.image.shape)
        age1 = np.zeros(image_shape, dtype=np.int8)

        xs0, ys0 = us0[mask, 0], us0[mask, 1]
        xs1, ys1 = us1[mask, 0], us1[mask, 1]
        age1[ys1, xs1] = age0[ys0, xs0] + 1

        age0 = age1


    plt.subplot(2, 2, 1)
    plt.title(f"frame {len(frames)-1}")
    plt.axis("off")
    plt.imshow(frames[-1].image, cmap="gray")

    plt.subplot(2, 2, 2)
    plt.title("History (the more bright, the older the pixel)")
    plt.axis("off")
    plt.imshow(age0, cmap="gray")

    for i in range(len(frames)-1):
        plt.subplot(4, 4, i+9)
        plt.title(f"frame {i}")
        plt.axis("off")
        plt.imshow(frames[i].image)

    plt.show()
