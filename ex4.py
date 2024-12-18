import cv2
from ultralytics import YOLO
import numpy as np
import sys
# YOLOv8モデルをロード
model = YOLO("yolov8x-pose.pt")


#取得したい画像の骨格取得
#選択したモデルに画像を突っ込み、結果を格納
results_1 = model("ex1.jpg", save=True, save_txt=True, save_conf=True)
keypoints_1 = results_1[0].keypoints


# ビデオファイルを開く
video_path = "ex3b.mp4"
cap = cv2.VideoCapture(video_path)

# リスト間の距離
def dist_frame(box1, box2):
    return np.linalg.norm(box1 - box2)

#画像の骨格座標を格納
p_1 = np.array([[float(x), float(y)]for x, y, _ in keypoints_1.data[0]])


        
frame_count = 0
match_frame = 0
#pythonでの最大値
dist  = sys.float_info.max
# ビデオフレームをループする
while cap.isOpened():
    # ビデオからフレームを読み込む
    success, frame = cap.read()

    if success:
        #Keypointsを取得
        results_2 = model(frame, save=True, save_txt=True, save_conf=True)
        keypoints_2 = results_2[0].keypoints

        #フレーム番号加算
        frame_count += 1

        #骨格データを座標配列pに格納
        p_2 = np.array([[float(x), float(y)]for x, y, _ in keypoints_2.data[0]])

        #距離小さければ記録
        temp = dist_frame(p_1, p_2)
        if( temp < dist):
            match_frame = frame_count
            dist = temp
        print(dist, match_frame)
        
        #経過フレーム表示
        if frame_count % 10 == 0:
            print(frame_count, "フレーム経過")
    else:
        # ビデオの終わりに到達したらループから抜ける
        break
cap.release()
print("一致したフレーム番号：", match_frame)