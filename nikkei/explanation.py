# ======================== カメラ姿勢の初期化 ========================
import numpy as np
from pathlib import Path
from skimage.io import imread

from tadataka.camera import CameraParameters, CameraModel
from tadataka.camera_distortion import FOV
from tadataka.features import extract_features, Features, Matcher
from tadataka.plot import plot_matches, plot_map
from tadataka.point_keypoint_map import subscribe, get_indices
from tadataka.pose import estimate_pose_change, Pose, solve_pnp
from tadataka.triangulation import Triangulation


filenames = sorted(Path("./datasets/nikkei/").glob("*.jpg"))
filenames = filenames[220:]

# 視差を十分に得るためにファイルを飛ばす
filenames = [filenames[0]] + filenames[4:]

# カメラ歪みや焦点距離等の補正のためにカメラのパラメータを与える
camera_parameters = CameraParameters(
    focal_length=[2890.16, 3326.04],
    offset=[1640, 1232]
)
distortion_model = FOV(0.01)  # カメラの歪みを表す
camera_model = CameraModel(camera_parameters, distortion_model)

# 特徴点のマッチングを行うオブジェクト
# 'match' は関数のように振る舞うことができる
match = Matcher()

# 画像を読み込む
image0 = imread(filenames[0])
image1 = imread(filenames[1])

# 特徴点を抽出する
keypoints0, descriptors0 = extract_features(image0)
keypoints1, descriptors1 = extract_features(image1)

# カメラ歪みの補正
keypoints0_undistorted = camera_model.undistort(keypoints0)
keypoints1_undistorted = camera_model.undistort(keypoints1)

features0 = Features(keypoints0_undistorted, descriptors0)
features1 = Features(keypoints1_undistorted, descriptors1)

# 特徴点のマッチングを行う
# 特徴点の座標と記述子の両方が必要なのでどちらも渡す
matches01 = match(features0, features1)

plot_matches(image0, image1, keypoints0, keypoints1, matches01)

# pose0 をカメラの基準座標とし，そこからの姿勢変化を推定していく
pose0 = Pose.identity()
# 基準座標からどれだけ変化したかを推定する
pose1 = estimate_pose_change(
    keypoints0_undistorted,
    keypoints1_undistorted,
    matches01
)

# ==================== カメラ姿勢の初期化ここまで ====================

# ========================== 3次元点の復元 ==========================

# カメラの姿勢が推定できたので，triangulationによって3次元点を復元する

t = Triangulation(pose0, pose1,
                  keypoints0_undistorted, keypoints1_undistorted)
# depth_mask はそれぞれの点がカメラの前にあるかどうかを表す配列
# カメラの後ろにあるものが撮影されることはありえないので，
# depth_mask を見ることで復元結果が正しいかどうかを見極めることができる
point_array01, depth_mask = t.triangulate(matches01)
# カメラの前にある点と，その点を作るのに使われた特徴点だけ残す
print(depth_mask)
point_array01 = point_array01[depth_mask]
matches01 = matches01[depth_mask]

# 地図を表示する
# plot_map_([pose0, pose1], points)

# 1フレーム目の特徴点，2フレーム目の特徴点，それらを使って復元された3次元点
# の対応関係を保存しておく
points, correspondence0, correspondence1 = subscribe(
    point_array01, matches01
)

plot_map([pose0, pose1], point_array01)

# ======================= 3次元点の復元ここまで =======================

# ======================= 3フレーム目の姿勢推定 =======================
image2 = imread(filenames[2])

keypoints2, descriptors2 = extract_features(image2)
keypoints2_undistorted = camera_model.undistort(keypoints2)
features2 = Features(keypoints2_undistorted, descriptors2)

matches02 = match(features0, features2)

plot_matches(image0, image2, keypoints0, keypoints2, matches02)

# 3フレーム目の姿勢推定では，1フレーム目と2フレーム目から
# 復元された3次元点 'point_array01' と，3フレーム目の特徴点 'keypoints2'
# の対応関係を得る必要がある．
# 3次元点は descriptor を持たないため，3次元点 'point_array01' と
# 3フレーム目の特徴点 'keypoints2' を直接マッチングすることはできない．
# したがって，1フレーム目(あるいは2フレーム目)の特徴点と
# 3フレーム目の特徴点のマッチングを行うことで，
# 3フレーム目の特徴点と3次元点の対応関係を得る．

# 3次元点 -> 1フレーム目の特徴点 の対応関係は既知
# 1フレーム目の特徴点 -> 3フレーム目の特徴点 の対応関係を得る
# 3次元点 -> 3フレーム目の特徴点 の対応関係を得る

# 1フレーム目と3フレーム目の対応関係を使って，3次元点と3フレーム目の特徴点の対応関係を得る
point_indices2, keypoint_indices2 = get_indices(correspondence0, matches02)

pose2 = solve_pnp(np.array([points[i] for i in point_indices2]),
                  keypoints2_undistorted[keypoint_indices2])

# =================== 3フレーム目の姿勢推定ここまで ===================
# ================== 3フレーム目による triangulation ==================

t = Triangulation(pose0, pose2,
                  keypoints0_undistorted, keypoints2_undistorted)

point_array02, depth_mask = t.triangulate(matches02)
point_array02 = point_array02[depth_mask]
matches02 = matches02[depth_mask]

plot_map([pose0, pose1, pose2],
         np.vstack((point_array01, point_array02)))

# ============= 3フレーム目による triangulation ここまで =============
