from skimage.io import imread
from pathlib import Path

from tadataka.camera import CameraParameters
from tadataka.camera.distortion import FOV
from tadataka.camera.io import load
from tadataka.plot import plot_map
from tadataka.visual_odometry import FeatureBasedVO


camera_models = load("./datasets/nikkei/cameras.txt")
camera_model = camera_models[1]
vo = FeatureBasedVO(camera_model, window_size=4)

filenames = sorted(Path("./datasets/nikkei/images").glob("*.jpg"))
filenames = filenames[70:]

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

    if i % 100 == 0:
        points, colors = vo.export_points()
        plot_map(vo.export_poses(), points, colors)

points, colors = vo.export_points()
plot_map(vo.export_poses(), points, colors)
