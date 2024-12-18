import cv2
from ultralytics import YOLO
import numpy as np

# YOLOv8モデルをロード
model = YOLO("yolov8x-pose.pt")

# 取得したい画像の骨格取得
# 選択したモデルに画像を突っ込み、結果を格納
results_1 = model("ex1.jpg")
keypoints_1 = results_1[0].keypoints



# 骨格座標を配列に変換
p_1 = [[int(x), int(y)] for x, y, _ in keypoints_1.data[0]]

# ビデオファイルを開く
video_path = "ex3b.mp4"
cap = cv2.VideoCapture(video_path)



# 座標比較に用いる許容誤差
threshold = 10

def is_close(point1, point2, threshold):
    """2つのポイントが閾値内で近いか確認"""
    return np.linalg.norm(np.array(point1) - np.array(point2)) <= threshold
    


def keypoints_match(box1, box2, threshold):
    """2つのキーポイントリストが一致するか確認"""
    if len(box1) != len(box2):
        return False
    return all(any(is_close(pt1, pt2, threshold) for pt2 in box2) for pt1 in box1)

frame_count = 0
found = False

# ビデオフレームをループする
while cap.isOpened():
    # ビデオからフレームを読み込む
    success, frame = cap.read()

    if success:
        
        frame_count += 1

        # Keypointsを取得
        results_2 = model(frame)
        keypoints_2 = results_2[0].keypoints

        if keypoints_2 is None:
            print(f"フレーム {frame_count}: keypoints が検出されませんでした。")
            continue

        # フレームの骨格座標を配列に変換
        p_2 = [[int(x), int(y)] for x, y, _ in keypoints_2.data[0]]

        # 一致チェック
        if keypoints_match(p_1, p_2, threshold):
            print("一致したフレーム番号：", frame_count)
            found = True
            break

        # デバッグ情報
        if frame_count % 10 == 0:
            print(f"フレーム {frame_count} を処理中...")
    else:
        break

if not found:
    print("一致するフレームが見つかりませんでした。")

cap.release()
