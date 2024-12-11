import cv2
from ultralytics import YOLO
import numpy as np

# YOLOv8モデルをロード
model = YOLO("yolov8x-pose.pt")


#取得したい画像の骨格取得
#選択したモデルに画像を突っ込み、結果を格納
results_1 = model("ex1.jpg", save=True, save_txt=True, save_conf=True)
keypoints_1 = results_1[0].keypoints


# ビデオファイルを開く
video_path = "ex3b.mp4"
cap = cv2.VideoCapture(video_path)

#骨格座標用の配列初期化
def position(box, x, y):
    box.append([int(x), int(y)])
    return 0

#リストが完全一致かどうかを判定
def list_match(box1, box2):
    if (len(box1) != len(box2)):
        return False
    return all(x == y and type(x) == type(y) for x, y in zip(box1, box2))

#画像の骨格座標を格納
p_1 = []
for x, y, _ in keypoints_1.data[0]:
    position(p_1, x, y)

frame_count = 0
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
        p_2 = []
        for x, y, _ in keypoints_2.data[0]:
            position(p_2, x, y)  
        
        #フレームと画像が一致ならばループから抜ける
        if (list_match(p_1, p_2) == True):
            print("一致したフレーム番号：", frame_count)
            break
        # 'q'が押されたらループから抜ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # ビデオの終わりに到達したらループから抜ける
        break
cap.release()