from pathlib import Path

from skimage.io import imread

from tadataka.camera.io import load
from tadataka.plot import plot_map
from tadataka.visual_odometry import FeatureBasedVO


camera_models = load("./datasets/nikkei/cameras.txt")
camera_model = camera_models[1]
vo = FeatureBasedVO(camera_model, window_size=4)

filenames = sorted(Path("./datasets/nikkei/images").glob("*.jpg"))

for i, filename in enumerate(filenames):
    # 画像の読み込み
    print("filename = {}".format(filename))
    image = imread(filename)

    # フレームの追加
    viewpoint = vo.add(image)

    # フレームの追加に失敗した場合負の数が返る
    if viewpoint < 0:
        continue

    # 不要なフレームを削除する
    vo.try_remove()

    # 20フレームごとに復元結果を表示する
    # ただし，1フレーム目が追加されたとき (i == 0) は
    # 3次元点が存在しないので表示を行わない
    if i > 0 and i % 20 == 0:
        points, colors = vo.export_points()
        plot_map(vo.export_poses(), points, colors)

points, colors = vo.export_points()
plot_map(vo.export_poses(), points, colors)
