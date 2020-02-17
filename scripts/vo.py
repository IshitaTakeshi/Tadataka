from pathlib import Path

from skimage.io import imread

from tadataka.camera.io import load
from tadataka.plot import plot_map
from tadataka.visual_odometry import FeatureBasedVO


# カメラパラメータの読み込み
camera_models = load("nikkei/cameras.txt")
# ID = 1 で登録されているパラメータを使う
# IDはカメラが複数あるときに使われるものなので今回は気にしなくてよい
camera_model = camera_models[1]
vo = FeatureBasedVO(camera_model, window_size=4)

filenames = sorted(Path("nikkei/images").glob("*.jpg"))

for i, filename in enumerate(filenames):
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
