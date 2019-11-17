from skimage.io import imread
from pathlib import Path

from tadataka.camera import CameraParameters
from tadataka.camera_distortion import FOV
from tadataka.plot import plot_map
from tadataka.visual_odometry import VisualOdometry


# saba
vo = VisualOdometry(
    CameraParameters(focal_length=[2890.16, 3326.04], offset=[1640, 1232]),
    FOV(0.01),
    max_active_keyframes=6
)

filenames = sorted(Path("./datasets/saba/").glob("*.jpg"))
filenames = [filenames[0]] + filenames[4:]

for i, filename in enumerate(filenames):
    print("filename = {}".format(filename))
    # 画像の読み込み
    image = imread(filename)

    # フレームの追加
    viewpoint = vo.add(image)

    # フレームの追加に失敗した場合負の数が返る
    if viewpoint < 0:
        continue

    # 不要なフレームを削除する
    vo.try_remove()

points, colors = vo.export_points()
plot_map(vo.export_poses(), points, colors)
