from skimage.io import imread
from pathlib import Path

from tadataka.camera import CameraParameters
from tadataka.camera_distortion import FOV
from tadataka.plot import plot_map
from tadataka.visual_odometry import VisualOdometry


# saba
vo = VisualOdometry(
    CameraParameters(focal_length=[3049, 4052], offset=[1640, 1232]),
    FOV(0.26),
    max_active_keyframes=8
)

filenames = sorted(Path("./datasets/nikkei/").glob("*.jpg"))
filenames = filenames[220:]
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

    if i == 0:
        continue

    # points, colors = vo.export_points()
    # plot_map(vo.export_poses(), points, colors)

points, colors = vo.export_points()
plot_map(vo.export_poses(), points, colors)
